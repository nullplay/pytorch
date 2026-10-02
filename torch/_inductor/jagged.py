"""
Packed jagged tensors in inductor (torch.ops.jagged.view / torch.ops.jagged.values, see torch/_jagged.py).

A jagged tensor x[B, J, *D] never exists densely. It lives in packed storage values[NNZ, *D] plus offsets[B + 1]:

    x[b, j, *d]  ->  values[off[b] + j, *d]        valid for j < off[b + 1] - off[b]

The IR keeps J as an ordinary size symbol, so pointwise / reduction / fusion treat it like any other dynamic dim.
Only the indexer (JaggedLayout: off[b] instead of J*b) and the loop bounds (J -> off[b + 1] - off[b]) know it varies
per row. Every realized buffer whose shape contains J is stored packed, the same way as its input.

Assumes the jagged dim is 1 and it depends on dim 0 (ShapeEnv.jagged_symbols[J] == (0, NNZ)).

Each fused kernel gets a strategy from its (numel, rnumel) group, right after fusion (plan_jagged_kernels):
  * J pointwise (J in numel): flatten. Each node's loop is reindexed so J merges with its parent dim b (wherever the
    two sit in the loop order) into packed rows p = off[b] + j. If b is still needed (e.g. a [B, D] broadcast), it
    is recovered with a binary search b = bucketize(p, off, right=True) - 1 (JaggedRow).
  * J in rnumel (reduced over J, or J pointwise with POINTWISE_STRATEGY = "loop"): jagged loop. J is the outermost
    reduction dim and each lane loops off[b + 1] - off[b] rows (JaggedLoop), b = the lane's row (x0 for numel == B,
    which is also XBLOCK = 1 / p0 only; x // D for numel == B * D, ...).
Codegen then only rewrites index expressions (rewrite_index: off[b] -> load, JaggedRow -> search) and the reduction
tree bound; tiling, splitting, masks and loops are the stock SIMD / Triton ones.
"""

import os
from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING

import sympy

import torch
import torch._jagged
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import SymPyValueRangeAnalysis, ValueRanges

from . import ir
from .loop_body import LoopBody
from .lowering import register_lowering
from .runtime.hints import AutotuneHint
from .utils import sympy_index_symbol, sympy_product, sympy_subs
from .virtualized import V


if TYPE_CHECKING:
    from .codegen.simd import SIMDKernel
    from .codegen.triton import TritonKernel
    from .scheduler import BaseSchedulerNode, SchedulerNode


def _offsets_name(J: sympy.Expr) -> str:
    jagged = getattr(V.graph, "jagged", None) if V.graph is not None else None
    return jagged[J][0] if jagged and J in jagged else f"off_{J}"


class JaggedOffset(sympy.Function):
    """off[b] of jagged symbol J: first packed row of batch b. Printed as <offsets buffer>[b]."""

    is_integer = True
    is_nonnegative = True

    def _torch_sympystr(self, printer: sympy.printing.StrPrinter) -> str:
        J, b = self.args
        return f"{_offsets_name(J)}[{printer._print(b)}]"

    _sympystr = _torch_sympystr


# value range analysis (bound_sympy) dispatches sympy functions by _torch_handler_name
JaggedOffset._torch_handler_name = "jagged_offset"
SymPyValueRangeAnalysis.jagged_offset = staticmethod(
    lambda J, b: ValueRanges(0, int_oo)
)


class JaggedRow(sympy.Function):
    """p0 of packed row x of jagged symbol J: the b with off[b] <= x < off[b + 1]. Only flattened kernels use it;
    the Triton kernel computes it once per program (binary search on offsets) outside any reduction loop."""

    is_integer = True
    is_nonnegative = True

    def _torch_sympystr(self, printer: sympy.printing.StrPrinter) -> str:
        J, x = self.args
        return f"bucketize({printer._print(x)}, {_offsets_name(J)})"

    _sympystr = _torch_sympystr


JaggedRow._torch_handler_name = "jagged_row"
SymPyValueRangeAnalysis.jagged_row = staticmethod(lambda J, x: ValueRanges(0, int_oo))


class JaggedLength(sympy.Function):
    """off[b + 1] - off[b]: what a bare J in an index means (arange(J), dense[:, :J], pad to N - J), the length of row
    b. plan_jagged_kernels substitutes it for J with b = the node's parent dim of J."""

    is_integer = True
    is_nonnegative = True

    def _torch_sympystr(self, printer: sympy.printing.StrPrinter) -> str:
        J, b = self.args
        return f"len({_offsets_name(J)}, {printer._print(b)})"

    _sympystr = _torch_sympystr


JaggedLength._torch_handler_name = "jagged_length"
SymPyValueRangeAnalysis.jagged_length = staticmethod(
    lambda J, b: ValueRanges(0, int_oo)
)
_JAGGED_FUNCTIONS = (JaggedOffset, JaggedRow, JaggedLength)


class JaggedLayout(ir.FixedLayout):
    """
    x[B, J, *D] over packed storage[NNZ, *D]: element (b, j, *d) is at
        offset + row_stride * (off[b] + j) + sum(d_i * inner_stride_i)
    stride[1:] = (row_stride, *inner_stride) are real; stride[0] = J * row_stride is only nominal.
    """

    strided = False  # ir.is_storage_and_layout: permute/expand/reshape views of it go through make_indexer

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: Sequence[sympy.Expr],
        row_stride: sympy.Expr,
        inner_stride: Sequence[sympy.Expr],
        offset: sympy.Expr,
        nnz: sympy.Expr,
    ) -> None:
        super().__init__(
            device,
            dtype,
            size,
            [size[1] * row_stride, row_stride, *inner_stride],
            offset,
        )
        self.nnz = nnz

    @classmethod
    def packed(cls, layout: ir.Layout, nnz: sympy.Expr) -> "JaggedLayout":
        inner = ir.FlexibleLayout.contiguous_strides(layout.size[1:])
        return cls(
            layout.device,
            layout.dtype,
            layout.size,
            inner[0],
            inner[1:],
            sympy.S.Zero,
            nnz,
        )

    def make_indexer(self) -> Callable[[Sequence[sympy.Expr]], sympy.Expr]:
        J, offset, row_stride, inner_stride = (
            self.size[1],
            self.offset,
            self.stride[1],
            self.stride[2:],
        )

        def indexer(index: Sequence[sympy.Expr]) -> sympy.Expr:
            b, j, *d = index
            return (
                offset
                + row_stride * (j + JaggedOffset(J, b))
                + sum((i * s for i, s in zip(d, inner_stride)), sympy.S.Zero)
            )

        return indexer

    def storage_size(self) -> sympy.Expr:
        return self.offset + self.nnz * self.stride[1]

    def __str__(self) -> str:
        return (
            f"JaggedLayout('{self.device.type}', {self.dtype}, size={list(self.size)}, "
            f"offsets={_offsets_name(self.size[1])}, nnz={self.nnz}, stride={list(self.stride[1:])})"
        )

    __repr__ = __str__


def dense_equivalent(index: sympy.Expr) -> sympy.Expr:
    """off[b] -> J*b: the [B, J] box index, for hints (stride order, tiling) where only access order matters."""
    return (
        index.replace(JaggedOffset, lambda J, b: J * b)
        .replace(JaggedRow, lambda J, x: FloorDiv(x, J))
        .replace(JaggedLength, lambda J, b: J)
    )


def offsets_reads(index: sympy.Expr) -> list[str]:
    """Offsets buffers an index reads through off[b]."""
    return [
        _offsets_name(o.args[0]) for o in sympy.sympify(index).atoms(*_JAGGED_FUNCTIONS)
    ]


def is_jagged(J: object) -> bool:
    return J in V.graph.jagged


def freeze_jagged_layouts(buffers: Sequence[ir.Buffer]) -> None:
    """Called before Scheduler: every buffer with J in its shape is stored packed, not as a [B, max J] box."""
    for op in buffers:
        if not isinstance(op, ir.ComputedBuffer):
            continue
        layout = op.get_layout()
        if isinstance(layout, JaggedLayout):
            continue
        size = layout.size
        if not any(sympy.sympify(s).free_symbols & V.graph.jagged.keys() for s in size):
            continue
        if (
            len(size) < 2
            or not is_jagged(size[1])
            or any(is_jagged(s) for s in [size[0], *size[2:]])
        ):
            raise NotImplementedError(
                f"jagged symbol outside dim 1 of {op.get_name()}: {size}"
            )
        op.layout = JaggedLayout.packed(layout, V.graph.jagged[size[1]][1])


@register_lowering(torch.ops.jagged.view, type_promotion_kind=None)
def jagged_view(values: ir.TensorBox, offsets: ir.TensorBox) -> ir.TensorBox:
    # values[NNZ, *D] reinterpreted as [B, J, *D]; no copy.
    J = V.graph.current_node.meta["val"].shape[1].node.expr
    offsets.realize()
    V.graph.jagged[J] = (offsets.get_name(), values.get_size()[0])
    size = [offsets.get_size()[0] - 1, J, *values.get_size()[1:]]
    # computed values: index them in place, packed row off[b] + j
    if not ir.is_storage_and_layout(values):
        loader = values.make_loader()
        return ir.Pointwise.create(
            device=values.get_device(),
            dtype=values.get_dtype(),
            inner_fn=lambda index: loader(
                [JaggedOffset(J, index[0]) + index[1], *index[2:]]
            ),
            ranges=size,
        )
    storage, layout = ir.as_storage_and_layout(values)
    return ir.TensorBox(
        ir.ReinterpretView(
            data=storage,
            layout=JaggedLayout(
                layout.device,
                layout.dtype,
                size,
                layout.stride[0],
                layout.stride[1:],
                layout.offset,
                layout.size[0],
            ),
        )
    )


@register_lowering(torch.ops.jagged.values, type_promotion_kind=None)
def jagged_values(x: ir.TensorBox, offsets: ir.TensorBox) -> ir.TensorBox:
    # x[B, J, *D] is already packed (a jagged view, or a buffer frozen to JaggedLayout): view its storage as [NNZ, *D].
    J = x.get_size()[1]
    x.realize()
    if isinstance(x.data, ir.ReinterpretView):
        storage, layout = x.data.data, x.data.layout
        if not isinstance(layout, JaggedLayout):
            raise NotImplementedError(f"jagged.values of a non-jagged view: {layout}")
    else:
        storage = x.data
        buf = storage.data if isinstance(storage, ir.StorageBox) else None
        if not isinstance(buf, ir.Buffer):
            raise NotImplementedError(f"jagged.values of {x}")
        layout = buf.get_layout()
        if not isinstance(layout, JaggedLayout):
            layout = buf.layout = JaggedLayout.packed(layout, V.graph.jagged[J][1])
    return ir.TensorBox(
        ir.ReinterpretView(
            data=storage,
            layout=ir.FixedLayout(
                layout.device,
                layout.dtype,
                [layout.nnz, *layout.size[2:]],
                layout.stride[1:],
                layout.offset,
            ),
        )
    )


# ---------------------------------------------------------------------------------------------------------------------
# codegen
#
# Kernel strategy is decided per fused kernel, from its (numel, rnumel) group, after fusion and any loop reordering
# (plan_jagged_kernels). After that the only jagged-specific codegen is in index expressions (rewrite_index) and in
# the bound of a reduction tree over J (JaggedLoop); everything else is the stock SIMD / Triton path.

# J pointwise: "flatten" (packed rows, default) or "loop" (J as a jagged inner loop, numel = the other dims)
POINTWISE_STRATEGY = os.environ.get("TORCHINDUCTOR_JAGGED_POINTWISE", "flatten")


def _has_jagged(e: sympy.Expr | Sequence[sympy.Expr]) -> bool:
    es = e if isinstance(e, (list, tuple)) else [e]
    return any(sympy.sympify(x).free_symbols & V.graph.jagged.keys() for x in es)


def _jagged_symbols(e: sympy.Expr) -> list[sympy.Symbol]:
    return [s for s in V.graph.jagged if s in sympy.sympify(e).free_symbols]


def _depend_size(J: sympy.Symbol) -> sympy.Expr:
    """B: number of rows of J (offsets has B + 1 entries)."""
    return V.graph.get_buffer(V.graph.jagged[J][0]).get_size()[0] - 1


def jagged_loop(rnumel: sympy.Expr) -> sympy.Symbol | None:
    """J if a kernel with this reduction numel loops over the jagged dim (JaggedLoop). plan_jagged_kernels has
    already flattened every other jagged dim and rejected kernels looping over more than one."""
    Js = _jagged_symbols(rnumel)
    return Js[0] if Js else None


def plan_jagged_kernels(nodes: Sequence["BaseSchedulerNode"]) -> None:
    """Scheduler pass after fusion (and loop reordering), before merge_loops. Per kernel, for each jagged dim J:
      * J reduced but its rows b are not (rnumel = f(J) * K, e.g. Min(J, 4096)): reorder pointwise nodes so J is the
        outermost reduction dim; the kernel then loops J with per-lane bound off[b + 1] - off[b] (JaggedLoop). At
        most one such J per kernel.
      * J pointwise, POINTWISE_STRATEGY == "loop": same jagged loop, J moved innermost and the kernel regrouped as
        (numel / J, J).
      * otherwise (J pointwise, or J reduced together with its rows): flatten J with its parent dim into packed
        rows (_flatten_node), before the jagged loop of another J is planned.
    merge_loops never merges J with another dim (SizeVarAllocator._simplify_loops), so either form survives it."""
    from .scheduler import FusedSchedulerNode, SchedulerNode

    for node in nodes:
        if type(node) not in (SchedulerNode, FusedSchedulerNode) or not node.is_gpu():
            continue
        snodes = [
            n
            for n in node.get_nodes()
            if isinstance(n, SchedulerNode) and not n.is_template()
        ]
        if not snodes:
            continue
        for n in snodes:
            _row_lengths(n)
        device, (numel, rnumel) = max(snodes, key=lambda n: int(n.is_reduction())).group
        Js = _jagged_symbols(sympy_product([numel, rnumel]))
        if not Js:
            continue
        # a reduction over both J and its rows (a total over a jagged tensor) flattens like a pointwise J
        red = [n for n in snodes if n.is_reduction()]
        loop = [
            J
            for J in _jagged_symbols(rnumel)
            if not any(_reduces_rows(n, J) for n in red)
        ]
        if len(loop) > 1:
            raise NotImplementedError(
                f"kernel {node.get_name()}: more than one jagged reduction dim {loop}"
            )
        if not loop and len(Js) == 1 and rnumel == 1 and POINTWISE_STRATEGY == "loop":
            (J,) = Js
            loop_numel = sympy.simplify(numel / J)
            if not _has_jagged(loop_numel) and all(J in n._sizes[0] for n in snodes):
                for n in snodes:
                    _order_for_jagged_loop(n, J, J)
                    n.group = (device, (loop_numel, J))
                continue
        # pointwise jagged dims flatten into packed rows; then a jagged reduction dim (one per kernel) is looped
        for J in Js:
            if J not in loop:
                for n in snodes:
                    _flatten_node(n, J)
        for J in loop:
            for n in snodes:
                _order_for_jagged_loop(n, J, rnumel)
        if isinstance(node, FusedSchedulerNode):
            node.group = max(snodes, key=lambda n: int(n.is_reduction())).group


def _row_lengths(node: "SchedulerNode") -> None:
    """Bare J in node's indices (not inside off[b] / bucketize) -> JaggedLength(J, b), b = its parent dim: the length
    of the row each element is in. The bound of an indirect index into a jagged dim (gather) becomes a named
    indexing expr for this, so it follows the node's indices like the rest."""
    body = node._body
    exprs, shims = dict(body.indexing_exprs), {}
    for name, m in body.submodules.items():
        kw = getattr(getattr(m, "clone", None), "keywords", {})
        if isinstance(kw.get("size"), sympy.Expr) and _has_jagged(kw["size"]):
            exprs[f"{name}_bound"], shims[name] = kw["size"], kw
    replacements = {}
    for J in V.graph.jagged:
        if not any(J in _strip_jagged(e).free_symbols for e in exprs.values()):
            continue
        # the row of the off[b] the node reads J's storage through (any expression: a jagged view of a jagged view
        # reads off_J[off_C[b] + c]), else the loop dim J depends on
        rows = OrderedSet(
            o.args[1]
            for e in exprs.values()
            for o in sympy.sympify(e).atoms(JaggedOffset)
            if o.args[0] == J
        )
        if len(rows) == 1:
            (b,) = rows
        else:
            ij = [
                i
                for i, s in enumerate([*node._sizes[0], *node._sizes[1]])
                if _has_jagged(s)
            ]
            b = [*body.iter_vars, *body.reduce_vars][
                _parent_dim(node, J, ij[0] if len(ij) == 1 else None)
            ]
        for name, e in exprs.items():
            e = replacements.get(name, e)
            protect = {
                a: sympy.Dummy() for a in sympy.sympify(e).atoms(*_JAGGED_FUNCTIONS)
            }
            new = (
                sympy.sympify(e)
                .xreplace(protect)
                .xreplace({J: JaggedLength(J, b)})
                .xreplace({v: k for k, v in protect.items()})
            )
            if new != e:
                replacements[name] = new
    if not replacements:
        return
    node.apply_indexing_exprs(replacements)
    for name, kw in shims.items():
        node._body.submodules[name] = node._body.bind_set_indirect_shim(
            kw["var"], f"{name}_bound", kw["check"], kw["wrap_neg"]
        )


def _reduces_rows(node: "SchedulerNode", J: sympy.Symbol) -> bool:
    sizes, rsizes = node._sizes
    ij = [i for i, s in enumerate(rsizes) if J in sympy.sympify(s).free_symbols]
    return (
        len(ij) == 1
        and rsizes[ij[0]] == J
        and _parent_dim(node, J, len(sizes) + ij[0]) >= len(sizes)
    )


def _strip_jagged(e: sympy.Expr) -> sympy.Expr:
    return sympy.sympify(e).xreplace(
        dict.fromkeys(sympy.sympify(e).atoms(*_JAGGED_FUNCTIONS), sympy.S.Zero)
    )


def _order_for_jagged_loop(
    node: "SchedulerNode", J: sympy.Symbol, rnumel: sympy.Expr
) -> None:
    """Make node's loops split onto (numel, rnumel = J * K) with J the outermost reduction dim: reduction nodes must
    already reduce over (J, *K); pointwise nodes are reordered to (*rest, J, *K)."""
    sizes, rsizes = node._sizes
    if node.is_reduction():
        if (
            _has_jagged(sizes)
            or not rsizes
            or J not in sympy.sympify(rsizes[0]).free_symbols
            or _has_jagged(rsizes[1:])
        ):
            raise NotImplementedError(
                f"{node.get_name()}: jagged loop needs J outermost in reduction {node._sizes}"
            )
        return
    ij = [i for i, s in enumerate(sizes) if _has_jagged(s)]
    if not ij and node.group[1][1] == 1:  # an epilogue over the kernel's numel
        return
    if len(ij) != 1 or J not in sympy.sympify(sizes[ij[0]]).free_symbols:
        raise NotImplementedError(
            f"{node.get_name()}: jagged dim not a loop of its own: {node._sizes}"
        )
    (ij,) = ij
    K = sympy.simplify(rnumel / sizes[ij])
    others = [i for i in range(len(sizes)) if i != ij]
    # shortest suffix of the other dims with product K is the inner part of r
    for n in range(len(others) + 1):
        tail = others[len(others) - n :]
        if V.graph.sizevars.statically_known_equals(
            sympy_product([sizes[i] for i in tail]), K
        ):
            break
    else:
        raise NotImplementedError(
            f"{node.get_name()}: cannot split {sizes} onto reduction {rnumel}"
        )
    order = [i for i in others if i not in tail] + [ij] + tail
    if order != list(range(len(sizes))):
        node.apply_new_loop_order(order)


def _parent_dim(node: "SchedulerNode", J: sympy.Symbol, ij: int | None) -> int:
    """Loop dim b of node that J depends on, counting reduction dims after pointwise ones: the var v of the off[v]
    its indices read (any position, so transposed or reordered loops are fine), else the only other dim of size B."""
    sizes = [*node._sizes[0], *node._sizes[1]]
    body = node._body
    iter_vars = [*body.iter_vars, *body.reduce_vars]
    found = OrderedSet[int]()
    for e in body.indexing_exprs.values():
        for o in sympy.sympify(e).atoms(JaggedOffset):
            if o.args[0] == J and o.args[1] in iter_vars:
                found.add(iter_vars.index(o.args[1]))
    if len(found) == 1:
        return next(iter(found))
    B = _depend_size(J)
    same = [
        i
        for i, s in enumerate(sizes)
        if i != ij and V.graph.sizevars.statically_known_equals(s, B)
    ]
    if not found and len(same) == 1:
        return same[0]
    raise NotImplementedError(
        f"{node.get_name()}: parent dim of {J} in {sizes} is ambiguous ({found}, {same})"
    )


def _flatten_node(node: "SchedulerNode", J: sympy.Symbol) -> None:
    """(.., d_b, .., d_J, ..) -> (.., p, ..): J merges into its parent dim b (wherever both are in the loop order,
    both pointwise or both reduced), p over packed rows. b, J used only as off[b] + j become p; if b is used on its
    own (e.g. a [B, D] broadcast) it becomes JaggedRow(J, p), a binary search, and j becomes p - off[b]."""
    sizes, rsizes = node._sizes
    dims = [*sizes, *rsizes]
    ij = [i for i, s in enumerate(dims) if J in sympy.sympify(s).free_symbols]
    if not ij:
        return
    if len(ij) != 1 or dims[ij[0]] != J:
        raise NotImplementedError(
            f"flatten {node.get_name()}: jagged dim not a loop of its own: {node._sizes}"
        )
    (ij,) = ij
    ib = _parent_dim(node, J, ij)
    if (ib < len(sizes)) != (ij < len(sizes)):
        raise NotImplementedError(
            f"flatten {node.get_name()}: {J} and its parent dim are not both pointwise or reduced"
        )
    _, nnz = V.graph.jagged[J]
    old = node._body
    pos = [i for i in range(len(dims)) if i != ij]  # old dim of each new loop
    new_dims = [nnz if i == ib else dims[i] for i in pos]
    n = len(sizes) - int(ij < len(sizes))
    (iter_vars, reduce_vars), var_ranges = _index_vars(new_dims[:n], new_dims[n:])

    def old_index(
        p0: sympy.Expr, index: Sequence[sympy.Expr]
    ) -> list[list[sympy.Expr]]:
        out = [sympy.S.Zero] * len(dims)
        for i, v in zip(pos, index):
            out[i] = v
        out[ib], out[ij] = p0, out[ib] - JaggedOffset(J, p0)
        return [out[: len(sizes)], out[len(sizes) :]]

    # p0 is only needed if some index still uses it once off[p0] + p1 is rewritten to p
    probe = sympy.Dummy("p0", integer=True, nonnegative=True)
    probe_index = old_index(probe, [*iter_vars, *reduce_vars])
    needs_p0 = any(
        probe in e.free_symbols
        for e in old.indexing_from_args(probe_index, True).values()
    )

    def body(index: Sequence[sympy.Expr], rindex: Sequence[sympy.Expr]) -> object:
        index = [*index, *rindex]
        p0 = JaggedRow(J, index[pos.index(ib)]) if needs_p0 else sympy.S.Zero
        return old(*old_index(p0, index), allow_same_symbol_in_index=True)

    new = LoopBody(body, [iter_vars, reduce_vars], var_ranges, iter_vars, reduce_vars)
    node._before_loop_state_mutation()
    node._body = new
    node._sizes = new.sizes
    device = node.group[0]
    node.group = (device, node.scheduler.get_backend(device).group_fn(node._sizes))
    node.refresh_dependencies(normalize=False, need_clear_tiling_cache=True)


def _index_vars(
    sizes: Sequence[sympy.Expr], rsizes: Sequence[sympy.Expr]
) -> tuple[
    tuple[list[sympy.Symbol], list[sympy.Symbol]], dict[sympy.Symbol, sympy.Expr]
]:
    iter_vars = [sympy_index_symbol(f"p{i}") for i in range(len(sizes))]
    reduce_vars = [sympy_index_symbol(f"p{len(sizes) + i}") for i in range(len(rsizes))]
    return (iter_vars, reduce_vars), dict(
        zip([*iter_vars, *reduce_vars], [*sizes, *rsizes])
    )


def rewrite_index(kernel: "SIMDKernel[Any]", index: sympy.Expr) -> sympy.Expr:
    """SIMDKernel.prepare_indexing hook. JaggedRow(J, x) -> binary search on offsets, JaggedOffset(J, b) -> load of
    off[b]. Both are emitted once in the kernel body, before any reduction loop: x and b may only depend on
    pointwise (x/y/z) range-tree symbols, like the entries inductor lifts out of the loop."""
    if not index.has(*_JAGGED_FUNCTIONS):
        return index
    from .codegen.triton import TritonKernel

    if not isinstance(kernel, TritonKernel):
        raise NotImplementedError(
            f"jagged tensors need the Triton backend, got {type(kernel)}"
        )
    index = index.replace(JaggedRow, lambda J, x: _row(kernel, J, x))

    def offset(J: sympy.Expr, b: sympy.Expr) -> sympy.Expr:
        start = _load_offset(kernel, J, b)
        if (loop := getattr(kernel, "jagged", None)) is not None and loop.J == J:
            loop.bind(b, start)
        return start

    index = index.replace(
        JaggedLength, lambda J, b: _load_offset(kernel, J, b + 1) - offset(J, b)
    )
    return index.replace(JaggedOffset, offset)


def _body_index(kernel: "TritonKernel", e: sympy.Expr) -> tuple[str, OrderedSet[str]]:
    """e printed in xindex/yindex terms, so the line does not depend on the x0 = ... entry lines having been emitted
    yet, and the masks of the trees it uses. A tree without a tensor dim (no_x_dim: XBLOCK = 1) is the scalar
    xoffset, so e is a scalar too."""
    from .codegen.triton import TritonSymbols

    syms = [s for s in e.free_symbols if s in kernel.range_tree_nodes]
    if any(TritonSymbols.is_reduction_index_symbol(kernel, s) for s in syms):
        raise NotImplementedError(f"jagged index {e} depends on a reduction index")
    masks, subs = OrderedSet[str](), {}
    for s in syms:
        entry = kernel.range_tree_nodes[s]
        if entry.root.tensor_dim is None:
            subs[s] = sympy_subs(
                entry.expr, {entry.root.index_sym(): entry.root.block_offset()}
            )
        else:
            subs[s] = entry.expr
            masks.add(entry.root.mask_name())
    return kernel.kexpr(kernel.rename_indexing(sympy_subs(e, subs))), masks


def _load_offset(kernel: "TritonKernel", J: sympy.Expr, b: sympy.Expr) -> sympy.Symbol:
    from .codegen.triton import TritonCSEVariable, TritonSymbols

    b_str, masks = _body_index(kernel, b)
    load_masks = OrderedSet(masks)
    kernel.filter_masks(load_masks)
    mask = f", {' & '.join(load_masks)}, other=0" if load_masks else ""
    var = kernel.cse.generate(
        kernel.body,
        f"tl.load({kernel.args.input(V.graph.jagged[J][0])} + ({b_str}){mask})",
        dtype=torch.int64,
        shape=TritonSymbols.get_block_shape(b),
    )
    if not isinstance(var, TritonCSEVariable):
        raise AssertionError(f"expected TritonCSEVariable, got {type(var)}")
    var.mask_vars = masks  # indices using it are masked like b
    return sympy_index_symbol(str(var))


def _row(kernel: "TritonKernel", J: sympy.Expr, x: sympy.Expr) -> sympy.Symbol:
    from .codegen.triton import TritonCSEVariable, TritonSymbols

    x_str, masks = _body_index(kernel, x)
    off, size = (
        kernel.args.input(V.graph.jagged[J][0]),
        kernel.index_to_str(_depend_size(J) + 1),
    )
    kernel.autotune_hints.add(AutotuneHint.ONE_ELEMENT_PER_THREAD)
    var = kernel.cse.generate(
        kernel.body,
        f"triton_helpers.bucketize_binary_search({x_str}, {off}, {size}, {size}, 1, 0, tl.int64, True, None, None, "
        "None) - 1",
        dtype=torch.int64,
        shape=TritonSymbols.get_block_shape(x),
    )
    if not isinstance(var, TritonCSEVariable):
        raise AssertionError(f"expected TritonCSEVariable, got {type(var)}")
    var.mask_vars = masks
    return sympy_index_symbol(str(var))


class JaggedLoop:
    """A TritonKernel whose reduction tree contains J (numel = J * K). J is off[b + 1] - off[b] for the row b of each
    lane; b is the pointwise expression of the first off[b] the kernel indexes (x0 for a [B] kernel, x // D for
    [B, D], ...), so the bound is emitted then, before the loop:
        extent   = (off[b + 1] - off[b]) * K
        r0_numel = extent              if b is a scalar (XBLOCK = 1 on the rows: one row per program)
                   tl.max(extent)      if b is a block (lanes of one program span rows of different lengths)
        r0_mask  = r0_index < extent
    All indices of one kernel must use the same b."""

    def __init__(self, kernel: "TritonKernel", J: sympy.Symbol) -> None:
        self.kernel, self.J = kernel, J
        (self.tree,) = [
            t for t in kernel.range_trees if t.is_reduction and _has_jagged(t.numel)
        ]
        self.b: sympy.Expr | None = None
        self.extent: str | None = None

    def bind(self, b: sympy.Expr, start: sympy.Symbol) -> None:
        if self.b is not None:
            if b != self.b:
                raise NotImplementedError(
                    f"jagged loop over {self.J} with rows {self.b} and {b}"
                )
            return
        from .codegen.triton import TritonSymbols

        kernel, self.b = self.kernel, b
        end = _load_offset(kernel, self.J, b + 1)
        extent = sympy_subs(self.tree.numel, {self.J: end - start})
        shape = TritonSymbols.get_block_shape(b)
        self.extent = str(
            kernel.cse.generate(
                kernel.body,
                kernel.kexpr(kernel.rename_indexing(extent)),
                dtype=torch.int64,
                shape=shape,
            )
        )
        numel = f"tl.max({self.extent})" if shape else self.extent
        kernel.body.writeline(
            f"{self.tree.prefix}numel = {numel}.to({kernel.index_dtype})"
        )
        kernel.codegen_reduction_numels(kernel.body)

    def mask_bound(self) -> str:
        if self.extent is None:
            raise NotImplementedError(
                f"jagged loop over {self.J}: no index reads off[b]"
            )
        return self.extent
