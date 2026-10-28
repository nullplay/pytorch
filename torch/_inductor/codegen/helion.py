from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, cast

import sympy

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from .. import config
from ..ir import IRNode, MutableBox, MutationLayoutSHOULDREMOVE, ReinterpretView
from ..ops_handler import DefaultHandler
from ..scheduler import SchedulerNode
from ..sizevars import SimplifyIndexing
from ..utils import IndentedBuffer
from ..virtualized import V
from .common import PythonPrinter
from .simd import SIMDKernel
from .simd_kernel_features import DisableReduction, EnableReduction, SIMDKernelFeatures


@dataclass(frozen=True)
class HelionValue:
    name: str
    depending_indices: tuple[str, ...]


@dataclass(frozen=True)
class HelionTensorArg:
    name: str
    operand: IRNode

    @property
    def source(self) -> str:
        return self.operand.get_name()


class HelionKernel(DefaultHandler):
    """Emit a source-only preview of MM/BMM schedules and pointwise epilogues."""

    def __init__(
        self, features: SIMDKernelFeatures, tiling: dict[str, sympy.Expr]
    ) -> None:
        if tuple(tiling) not in (("y", "x", "r0_"), ("z", "y", "x", "r0_")) or not all(
            isinstance(size, sympy.Integer) for size in tiling.values()
        ):
            raise NotImplementedError("Helion preview requires static MM/BMM tiling")
        if not config.triton.codegen_upcast_to_fp32:
            raise NotImplementedError("Helion preview requires FP32 compute upcasting")
        nodes = tuple(features.scheduler_nodes())
        self.matmul_inputs = {
            name: V.graph.helion_matmul_inputs[name]
            for node in nodes
            for name in node.get_buffer_names()
            if name in V.graph.helion_matmul_inputs
        }
        if len(self.matmul_inputs) != features.op_counts().get("dot", 0):
            raise NotImplementedError("Helion preview requires lowered MM/BMM operands")
        if any(
            operand.get_dtype() not in (torch.float16, torch.bfloat16, torch.float32)
            for operands in self.matmul_inputs.values()
            for operand in operands
        ):
            raise NotImplementedError("Helion preview supports FP16/BF16/FP32 matmul")
        self.features = features
        self.tiling = tiling
        self.outer_axes = tuple(tiling)[:-1]
        self.symbols = {
            axis: sympy.Symbol(f"helion_{axis}", integer=True) for axis in tiling
        }
        self.tiles = {"z": "tile_z", "y": "tile_y", "x": "tile_x", "r0_": "tile_r"}
        self.accumulators = {
            name: "acc" if number == 0 else f"acc{number}"
            for number, name in enumerate(self.matmul_inputs)
        }
        self.current_matmul: str | None = None
        self.inputs: OrderedSet[str] = OrderedSet()
        fused_names = OrderedSet(node.get_name() for node in nodes)
        self.outputs = OrderedSet(
            output.get_name()
            for node in nodes
            for output in node.get_outputs()
            if not isinstance(output.node.get_layout(), MutationLayoutSHOULDREMOVE)
            and not V.graph.scheduler.can_buffer_be_removed_through_fusion(
                output.get_name(), fused_names
            )
        )
        self.arguments: dict[tuple[Any, ...], HelionTensorArg] = {}
        self.store_forwarding: dict[tuple[str, sympy.Expr], HelionValue] = {}
        self.indirect_indices: dict[sympy.Symbol, HelionValue] = {}
        self.reduction_body = IndentedBuffer()
        self.epilogue = IndentedBuffer()
        self.body = self.reduction_body
        # A tile_r temporary cannot be reused outside the K loop.
        self.cse: dict[
            IndentedBuffer, dict[tuple[str, tuple[str, ...]], HelionValue]
        ] = {self.reduction_body: {}, self.epilogue: {}}
        self.counter = itertools.count()
        self.dots_emitted: OrderedSet[str] = OrderedSet()

    @staticmethod
    def compute_dtype(dtype: torch.dtype) -> torch.dtype:
        if config.triton.codegen_upcast_to_fp32 and dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            return torch.float32
        return dtype

    def emit(self, expression: str, depending_indices: tuple[str, ...]) -> HelionValue:
        cache = self.cse[self.body]
        key = (expression, depending_indices)
        if key not in cache:
            name = f"tmp{next(self.counter)}"
            self.body.writeline(f"{name} = {expression}")
            cache[key] = HelionValue(name, depending_indices)
        return cache[key]

    def align(self, value: HelionValue, depending_indices: tuple[str, ...]) -> str:
        if not value.depending_indices or value.depending_indices == depending_indices:
            return value.name
        if not OrderedSet(value.depending_indices).issubset(depending_indices):
            raise NotImplementedError("Helion preview cannot drop a tensor axis")
        order = tuple(
            axis for axis in depending_indices if axis in value.depending_indices
        )
        result = value.name
        if order != value.depending_indices:
            permutation = [value.depending_indices.index(axis) for axis in order]
            result += f".permute({', '.join(map(str, permutation))})"
        if len(depending_indices) != len(value.depending_indices):
            shape = [
                self.tiles[axis] if axis in value.depending_indices else "1"
                for axis in depending_indices
            ]
            result += f".reshape([{', '.join(shape)}])"
        return result

    def tensor_arg(self, operand: IRNode, alias: str | None = None) -> HelionTensorArg:
        while isinstance(operand, MutableBox):
            operand = operand.data
        layout = operand.get_layout()
        key = (
            operand.get_name(),
            tuple(layout.size),
            tuple(layout.stride),
            layout.offset,
        )
        if key not in self.arguments:
            if alias is None:
                prefix = "out" if operand.get_name() in self.outputs else "in"
                number = sum(
                    arg.name.startswith(prefix) for arg in self.arguments.values()
                )
                alias = f"{prefix}{number}"
            self.arguments[key] = HelionTensorArg(alias, operand)
        return self.arguments[key]

    def tensor_indices(
        self, operand: IRNode, index: sympy.Expr
    ) -> tuple[list[str], tuple[str, ...]]:
        layout = operand.get_layout()
        # For shape (M, N), stride (N, 1), match N*y + x to [tile_y, tile_x].
        # A bias with shape (N,) and index x becomes bias[tile_x].
        candidates = [
            [
                axis
                for axis in self.outer_axes
                if V.graph.sizevars.statically_known_equals(size, self.tiling[axis])
            ]
            + ([None] if size == 1 else [])
            for size in layout.size
        ]
        for axes in itertools.product(*candidates):
            active = tuple(axis for axis in axes if axis is not None)
            if len(active) != len(OrderedSet(active)):
                continue
            expected = layout.offset + sum(
                self.symbols[axis] * stride
                for axis, stride in zip(axes, layout.stride)
                if axis is not None
            )
            if V.graph.sizevars.statically_known_equals(index, expected):
                return (
                    [self.tiles[axis] if axis is not None else "0" for axis in axes]
                    or ["()"],
                    active,
                )
        # A flattened MM row y can address a 3D output as [y//M, y%M, x].
        # Decode the existing layout; keep ordinary Tile loads above for TMA.
        order = sorted(range(len(layout.size)), key=lambda dim: layout.stride[dim])
        span = 1
        for dim in order:
            size, stride = layout.size[dim], layout.stride[dim]
            if not isinstance(size, (int, sympy.Integer)) or not isinstance(
                stride, (int, sympy.Integer)
            ):
                raise NotImplementedError("Helion indexing requires static layouts")
            if size == 1:
                continue
            if stride < span:
                raise NotImplementedError(
                    "Helion indexing requires nonoverlapping strides"
                )
            span += (size - 1) * stride
        offset = index - layout.offset
        coordinates = []
        var_ranges = dict(zip(self.symbols.values(), self.tiling.values()))
        for dim, (size, stride) in enumerate(zip(layout.size, layout.stride)):
            coordinate = FloorDiv(offset, stride)
            if dim != order[-1]:
                # Mod drops whole rows, including negative indirect row indices.
                coordinate = sympy.Mod(coordinate, size).replace(
                    sympy.Mod, lambda base, modulus: ModularIndexing(base, 1, modulus)
                )
            coordinates.append(
                sympy.S.Zero
                if size == 1
                else V.graph.sizevars.simplify_with_ranges(coordinate, var_ranges)
            )
        restored = sum(
            coord * stride for coord, stride in zip(coordinates, layout.stride)
        )
        if V.graph.sizevars.simplify_with_ranges(restored - offset, var_ranges) != 0:
            raise NotImplementedError(f"Cannot recover Helion tensor indices: {index}")
        symbols = OrderedSet().union(*(coord.free_symbols for coord in coordinates))
        axes = tuple(
            axis
            for axis in self.outer_axes
            if self.symbols[axis] in symbols
            or any(
                axis in self.indirect_indices[symbol].depending_indices
                for symbol in symbols
                if symbol in self.indirect_indices
            )
        )
        replacements = {
            symbol: sympy.Symbol(
                self.align(HelionValue(f"{self.tiles[axis]}.index", (axis,)), axes)
            )
            for axis, symbol in self.symbols.items()
            if symbol in symbols
        }
        replacements.update(
            {
                symbol: sympy.Symbol(self.align(value, axes))
                for symbol, value in self.indirect_indices.items()
                if symbol in symbols
            }
        )
        if symbols - replacements.keys():
            raise NotImplementedError(f"Unsupported Helion indexing: {index}")
        printer = PythonPrinter()
        return [
            printer.doprint(coord.xreplace(replacements), simplify=False)
            for coord in coordinates
        ], axes

    def indirect_indexing(
        self,
        value: HelionValue,
        size: sympy.Expr,
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        # out.index_add_(0, indices, acc) becomes hl.atomic_add(out, [indices, x], acc).
        # Helion's tensor indexing handles negative-index wrapping.
        # For wrap_neg=False, this preview assumes valid nonnegative indices;
        # it does not yet emit Inductor's runtime index checks.
        symbol = sympy.Symbol(
            f"helion_indirect{len(self.indirect_indices)}", integer=True
        )
        self.indirect_indices[symbol] = value
        return symbol

    def load(self, name: str, index: sympy.Expr) -> HelionValue:
        index = sympy.sympify(index)
        if (name, index) in self.store_forwarding:
            return self.store_forwarding[name, index]
        if self.current_matmul is not None:
            operands = self.matmul_inputs[self.current_matmul]
            number = tuple(self.matmul_inputs).index(self.current_matmul)
            batch = ("z",) if "z" in self.tiling else ()
            for operand, axes, alias in (
                (operands[0], (*batch, "y", "r0_"), "A"),
                (operands[1], (*batch, "r0_", "x"), "B"),
            ):
                if operand.get_name() != name:
                    continue
                expected = operand.make_indexer()([self.symbols[axis] for axis in axes])
                if V.graph.sizevars.statically_known_equals(index, expected):
                    argument = self.tensor_arg(
                        operand, alias + (str(number) if number else "")
                    )
                    self.inputs.add(name)
                    coordinates = ", ".join(self.tiles[axis] for axis in axes)
                    dtype = self.compute_dtype(operand.get_dtype())
                    return self.emit(
                        f"{argument.name}[{coordinates}].to({dtype})", axes
                    )
            raise NotImplementedError("Helion preview does not support fused prologues")
        operand = V.graph.get_buffer(name)
        coordinates, axes = self.tensor_indices(operand, index)
        argument = self.tensor_arg(operand)
        self.inputs.add(name)
        dtype = self.compute_dtype(operand.get_dtype())
        return self.emit(f"{argument.name}[{', '.join(coordinates)}].to({dtype})", axes)

    def constant(self, value: bool | float | int, dtype: torch.dtype) -> HelionValue:
        dtype = self.compute_dtype(dtype)
        if isinstance(value, float) and not math.isfinite(value):
            literal = f"float({str(value)!r})"
        else:
            literal = repr(value)
        return self.emit(f"hl.full([], {literal}, dtype={dtype})", ())

    def to_dtype(
        self,
        value: HelionValue,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> HelionValue:
        if use_compute_types:
            dtype = self.compute_dtype(dtype)
        return self.emit(f"{value.name}.to({dtype})", value.depending_indices)

    def dot(self, lhs: HelionValue, rhs: HelionValue) -> HelionValue:
        name = self.current_matmul
        if name is None or name in self.dots_emitted:
            raise NotImplementedError("Helion dot must belong to a lowered MM/BMM node")
        self.dots_emitted.add(name)
        batch = ("z",) if "z" in self.tiling else ()
        left = self.align(lhs, (*batch, "y", "r0_"))
        right = self.align(rhs, (*batch, "r0_", "x"))
        acc = self.accumulators[name]
        # Horizontal fusion needs one accumulator per MM, not one shared sum.
        op = "baddbmm" if batch else "addmm"
        self.body.writeline(f"{acc} = torch.{op}({acc}, {left}, {right})")
        return HelionValue(acc, self.outer_axes)

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: str,
        value: HelionValue,
    ) -> HelionValue:
        if reduction_type != "dot":
            raise NotImplementedError("Helion preview supports one dot reduction")
        return value

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: HelionValue,
        mode: str | None = None,
    ) -> None:
        if mode not in (None, "atomic_add"):
            raise NotImplementedError(f"Unsupported Helion store mode: {mode}")
        operand = V.graph.get_buffer(name)
        layout = operand.get_layout()
        if isinstance(layout, MutationLayoutSHOULDREMOVE):
            if mode is None:
                raise NotImplementedError(
                    "Helion preview supports atomic_add mutations"
                )
            operand = layout.target
            self.inputs.add(operand.get_name())
        if mode == "atomic_add" or name in self.outputs:
            # Physical stores run in the epilogue and may alias cached loads.
            self.cse[self.epilogue].clear()
        if mode == "atomic_add":
            if operand.get_dtype() == torch.bool:
                raise NotImplementedError(
                    "Helion preview does not support bool atomic stores"
                )
            coordinates, axes = self.tensor_indices(operand, index)
            argument = self.tensor_arg(operand)
            stored = self.align(value, axes)
            self.epilogue.writeline(
                f"hl.atomic_add({argument.name}, [{', '.join(coordinates)}], "
                f"{stored}.to({operand.get_dtype()}))"
            )
            return
        self.store_forwarding[name, sympy.sympify(index)] = value
        if name in self.outputs:
            coordinates, axes = self.tensor_indices(operand, index)
            argument = self.tensor_arg(operand)
            stored = self.align(value, axes)
            self.epilogue.writeline(
                f"{argument.name}[{', '.join(coordinates)}] = "
                f"{stored}.to({operand.get_dtype()})"
            )

    def store_reduction(self, name: str, index: sympy.Expr, value: HelionValue) -> None:
        self.store(name, index, value)

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        binary = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "truediv": "/",
            "eq": "==",
            "ne": "!=",
            "lt": "<",
            "le": "<=",
            "gt": ">",
            "ge": ">=",
        }
        unary = OrderedSet(["relu", "erf", "tanh", "neg", "sigmoid", "exp", "expm1"])
        torch_ops = OrderedSet(["where", "minimum", "maximum"])
        if not kwargs and (name in binary or name in unary or name in torch_ops):
            depending_indices = tuple(
                dict.fromkeys(
                    axis for value in args for axis in value.depending_indices
                )
            )
            values = [self.align(value, depending_indices) for value in args]
            # Torch pointwise ops are lowered by Helion; align the tile axes first.
            expression = (
                f"{values[0]} {binary[name]} {values[1]}"
                if name in binary
                else f"torch.{name}({', '.join(values)})"
            )
            return self.emit(expression, depending_indices)
        raise NotImplementedError(f"Unsupported Helion preview op: {name}")

    def codegen_kernel(self) -> str:
        """Replay the SIMD schedule and emit the kernel plus its metadata-view caller."""

        def set_ranges(*ranges):
            # SIMD may split y into batch/row: y//M and y%M, without another loop.
            return [
                [
                    FloorDiv(symbol, math.prod(sizes[dim + 1 :]))
                    if dim == 0
                    else ModularIndexing(symbol, math.prod(sizes[dim + 1 :]), size)
                    for dim, size in enumerate(sizes)
                ]
                for symbol, sizes in zip(self.symbols.values(), ranges)
            ]

        var_ranges = dict(zip(self.symbols.values(), self.tiling.values()))
        for node in self.features.node_schedule:
            if node is DisableReduction:
                self.body = self.epilogue
            elif node is EnableReduction:
                self.body = self.reduction_body
            else:
                node = cast(SchedulerNode, node)
                self.current_matmul = next(
                    (
                        name
                        for name in node.get_buffer_names()
                        if name in self.matmul_inputs
                    ),
                    None,
                )
                groups = list(self.tiling.values())
                if self.body is self.epilogue:
                    groups[-1] = sympy.S.One
                indices = SIMDKernel.map_kernel_groups_to_node_sizes(
                    groups, node.get_ranges(), set_ranges
                )
                with V.set_ops_handler(SimplifyIndexing(self, var_ranges)):
                    node._body(*indices)
        code = IndentedBuffer()
        code.writelines(
            ["import torch", "import helion", "import helion.language as hl", ""]
        )
        tensor_args = sorted(
            self.arguments.values(), key=lambda arg: arg.source in self.outputs
        )
        if any(isinstance(arg.operand, ReinterpretView) for arg in tensor_args):
            code.writeline(
                "from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor"
            )
            code.writeline("")
        code.writeline('@helion.kernel(static_shapes=True, dot_precision="ieee")')
        name = "helion_" + "_".join(
            node.get_name() for node in self.features.scheduler_nodes()
        )
        signature = ", ".join(f"{arg.name}: torch.Tensor" for arg in tensor_args)
        code.writeline(f"def {name}({signature}) -> None:")
        with code.indent():
            tiles = ", ".join(self.tiles[axis] for axis in self.outer_axes)
            sizes = ", ".join(str(self.tiling[axis]) for axis in self.outer_axes)
            code.writeline(f"for {tiles} in hl.tile([{sizes}]):")
            with code.indent():
                for acc in self.accumulators.values():
                    code.writeline(f"{acc} = hl.zeros([{tiles}], dtype=torch.float32)")
                code.writeline(f"for tile_r in hl.tile({self.tiling['r0_']}):")
                with code.indent():
                    code.splice(self.reduction_body)
                code.splice(self.epilogue)
        code.writeline("")
        signature = ", ".join(
            f"{arg}: torch.Tensor" for arg in (*self.inputs, *self.outputs)
        )
        code.writeline(f"def call_{name}({signature}) -> None:")
        with code.indent():
            # a.shape == (33, 9, 8) can be passed as an existing (33, 72) view.
            # Helion then loads A[tile_y, tile_r], without r//8 or r%8.
            arguments = ", ".join(
                arg.operand.codegen_reference(code)
                if isinstance(arg.operand, ReinterpretView)
                else arg.source
                for arg in tensor_args
            )
            code.writeline(f"{name}({arguments})")
        return code.getvalue()
