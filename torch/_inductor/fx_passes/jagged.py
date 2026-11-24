"""
Post-grad passes for jagged (variable-length) tensors.

lower_fbgemm_jagged_ops rewrites fbgemm_gpu's jagged ops into the aten jagged ops
that Inductor lowers to masked gathers (see jagged_lowerings.py), so they fuse with
their neighbors. For `jagged_to_padded_dense_forward(values * 2, [offsets], [T]) + 1`
the wrapper goes from an extern kernel between two pointwise kernels

    triton_poi_fused_mul_0.run(arg0_1, buf0, ...)
    buf1 = torch.ops.fbgemm.jagged_to_padded_dense_forward.default(buf0, [arg1_1], ...)
    triton_poi_fused_add_1.run(buf2, ...)

to one kernel that reads values[offsets[b] + t] where t < lengths[b], else padding:

    triton_poi_fused_add_mul_0.run(arg1_1, arg0_1, buf0, ...)

jagged_bmm_to_grouped_mm replaces a batched matmul on padded jagged rows,
`padded_dense_to_jagged(bmm(jagged_to_padded_dense(values), weight))`, with
`_grouped_mm(values, weight, offs)`, which multiplies only the real rows: B
padded GEMMs of T rows each become one grouped GEMM over the L jagged rows.
"""

from __future__ import annotations

import sys
from typing import Any

import torch
from torch._guards import detect_fake_mode
from torch.fx import Graph, Node
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.node import map_arg


aten = torch.ops.aten


def _schema_name(node: Node) -> str:
    target = node.target
    if node.op != "call_function" or not isinstance(target, torch._ops.OpOverload):
        return ""
    return target._schema.name


def _val(node: Any) -> Any:
    return node.meta.get("val") if isinstance(node, Node) else node


def _only_offsets(offsets: Any) -> Node | None:
    """The offsets of the jagged dimension; like the aten ops, only one is supported."""
    if isinstance(offsets, (list, tuple)) and len(offsets) == 1:
        return offsets[0]
    return None


def _arg(node: Node, index: int, name: str, default: Any = None) -> Any:
    args = node.args
    return args[index] if len(args) > index else node.kwargs.get(name, default)


class _Builder:
    """Inserts nodes before `anchor` and fills in their meta["val"]."""

    def __init__(self, graph: Graph, anchor: Node, fake_mode: Any) -> None:
        self.graph = graph
        self.anchor = anchor
        self.fake_mode = fake_mode

    def __call__(self, target: Any, *args: Any, **kwargs: Any) -> Node:
        with self.graph.inserting_before(self.anchor):
            node = self.graph.call_function(target, args, kwargs)
        fake_args, fake_kwargs = map_arg((args, kwargs), _val)
        with self.fake_mode:
            node.meta["val"] = target(*fake_args, **fake_kwargs)
        return node


def _complete_cumsum(b: _Builder, lengths: Node) -> Node:
    padded = b(aten.constant_pad_nd.default, lengths, [1, 0], 0)
    # fbgemm keeps int32 offsets int32; cumsum would promote them to int64.
    return b(aten.cumsum.default, padded, 0, dtype=_val(lengths).dtype)


def _to_padded(
    b: _Builder, values: Node, offsets: Node, max_length: Any, padding_value: float
) -> Node | None:
    # The aten op takes [L, D] values; [L] values (e.g. ids) go through [L, 1].
    rank = _val(values).dim()
    if rank not in (1, 2):
        return None
    if rank == 1:
        values = b(aten.unsqueeze.default, values, -1)
    dense = b(
        aten._jagged_to_padded_dense_forward.default,
        values,
        [offsets],
        [max_length],
        padding_value,
    )
    return b(aten.squeeze.dim, dense, -1) if rank == 1 else dense


def _to_jagged(b: _Builder, dense: Node, offsets: Node, total: Any) -> Node | None:
    if _val(dense).dim() != 3 or total is None:
        return None
    return b(aten._padded_dense_to_jagged_forward.default, dense, [offsets], total)


def _lower_fbgemm_op(b: _Builder, node: Node) -> Node | None:
    name = _schema_name(node)
    args: Any = node.args
    if name == "fbgemm::asynchronous_complete_cumsum":
        lengths = args[0]
        return _complete_cumsum(b, lengths) if _val(lengths).dim() == 1 else None
    if name == "fbgemm::pack_segments":
        t_in, lengths, max_length = args
        if _val(t_in).dim() not in (1, 2) or _val(lengths).dim() != 1:
            return None
        return _to_padded(b, t_in, _complete_cumsum(b, lengths), max_length, 0.0)
    if name == "fbgemm::pack_segments_backward":
        data, lengths, total_length = args[:3]
        if _val(data).dim() != 3 or _val(lengths).dim() != 1:
            return None
        return _to_jagged(b, data, _complete_cumsum(b, lengths), total_length)
    if name == "fbgemm::jagged_to_padded_dense_forward":
        offsets = _only_offsets(args[1])
        if offsets is None:
            return None
        padding_value = _arg(node, 3, "padding_value", 0.0)
        return _to_padded(b, args[0], offsets, args[2][0], padding_value)
    if name in (
        "fbgemm::jagged_to_padded_dense_backward",
        "fbgemm::dense_to_jagged_forward",
    ):
        offsets = _only_offsets(args[1])
        if offsets is None:
            return None
        return _to_jagged(b, args[0], offsets, _arg(node, 2, "total_L"))
    return None


def lower_fbgemm_jagged_ops(graph: Graph) -> None:
    fake_mode = detect_fake_mode([_val(n) for n in graph.find_nodes(op="placeholder")])
    if fake_mode is None:
        return
    for node in list(graph.nodes):
        if not _schema_name(node).startswith("fbgemm::"):
            continue
        new = _lower_fbgemm_op(_Builder(graph, node, fake_mode), node)
        if new is not None:
            node.replace_all_uses_with(new)
            graph.erase_node(node)


_VIEWS = (aten.view.default, aten.reshape.default, aten._unsafe_view.default)
# The bf16 CUDA _grouped_mm kernel (CUTLASS, sm90) takes fewer than 1024 groups.
_MAX_GROUPS = 1023


def _offsets_key(offsets: Node) -> Node:
    """Offsets [0, cumsum(lengths)] compare equal through their lengths, whether
    built as cumsum(pad(lengths)) or pad(cumsum(lengths))."""
    ops = (aten.cumsum.default, aten.constant_pad_nd.default)
    inner = offsets.args[0] if offsets.target in ops else None
    if not isinstance(inner, Node) or inner.target not in ops:
        return offsets
    if inner.target is offsets.target:
        return offsets
    pad_args: Any = (offsets if offsets.target is ops[1] else inner).args
    if list(pad_args[1]) != [1, 0] or (len(pad_args) > 2 and pad_args[2] != 0):
        return offsets
    lengths: Any = inner.args[0]
    return lengths


def _rewrite_jagged_bmm(graph: Graph, fake_mode: Any, bmm: Node) -> None:
    lhs, weight = bmm.args
    padded = lhs
    while isinstance(padded, Node) and padded.target in _VIEWS:
        padded = padded.args[0]
    out = bmm
    while len(out.users) == 1 and next(iter(out.users)).target in _VIEWS:
        out = next(iter(out.users))
    if len(out.users) != 1:
        return
    jagged = next(iter(out.users))
    if not (
        isinstance(padded, Node)
        and padded.target is aten._jagged_to_padded_dense_forward.default
        and jagged.target is aten._padded_dense_to_jagged_forward.default
    ):
        return
    pargs: Any = padded.args
    jargs: Any = jagged.args
    values, max_length, total = pargs[0], pargs[2][0], jargs[2]
    offsets, jagged_offsets = _only_offsets(pargs[1]), _only_offsets(jargs[1])
    if offsets is None or jagged_offsets is None:
        return
    if _offsets_key(offsets) is not _offsets_key(jagged_offsets):
        return
    (L, C), (B, M, K), N = _val(values).shape, _val(lhs).shape, _val(weight).shape[2]
    if not all(isinstance(d, int) for d in (C, K, N)) or C % K or K % 8 or N % 8:
        return
    H, T = C // K, _val(padded).shape[1]
    if not (
        _val(values).dtype is torch.bfloat16
        and _val(weight).dtype is torch.bfloat16
        and _val(values).is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(_val(values).device) >= (9, 0)
        and statically_known_true(B <= _MAX_GROUPS)
        and statically_known_true(M == T * H)
        and statically_known_true(_val(total) == L)
        and tuple(_val(out).shape[2:]) == (H * N,)
    ):
        return
    b = _Builder(graph, jagged, fake_mode)
    rows = b(aten.reshape.default, values, [-1, K])
    ends = b(aten.slice.Tensor, offsets, 0, 1, sys.maxsize)
    if H != 1:
        ends = b(aten.mul.Tensor, ends, H)
    ends = b(torch.ops.prims.convert_element_type.default, ends, torch.int32)
    result = b(aten._grouped_mm.default, rows, weight, ends)
    result = b(aten.reshape.default, result, [-1, H * N])
    # The original drops the rows of sequences longer than max_length and writes
    # zeros there; going through the padded layout does the same, and the pair
    # fuses into one masked copy.
    to_padded = aten._jagged_to_padded_dense_forward.default
    result = b(to_padded, result, [offsets], [max_length], 0.0)
    result = b(aten._padded_dense_to_jagged_forward.default, result, [offsets], total)
    jagged.replace_all_uses_with(result)
    graph.erase_node(jagged)


def jagged_bmm_to_grouped_mm(graph: Graph) -> None:
    fake_mode = detect_fake_mode([_val(n) for n in graph.find_nodes(op="placeholder")])
    if fake_mode is None:
        return
    for bmm in graph.find_nodes(op="call_function", target=aten.bmm.default):
        _rewrite_jagged_bmm(graph, fake_mode, bmm)
