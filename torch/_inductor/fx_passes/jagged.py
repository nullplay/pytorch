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
"""

from __future__ import annotations

from typing import Any

import torch
from torch._guards import detect_fake_mode
from torch.fx import Graph, Node
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
        if _val(lengths).dim() != 1:
            return None
        padded = b(aten.constant_pad_nd.default, lengths, [1, 0], 0)
        # fbgemm keeps int32 offsets int32; cumsum would promote them to int64.
        return b(aten.cumsum.default, padded, 0, dtype=_val(lengths).dtype)
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
