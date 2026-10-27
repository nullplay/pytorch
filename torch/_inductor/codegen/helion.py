from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, cast

import sympy

import torch
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..ops_handler import DefaultHandler
from ..scheduler import SchedulerNode
from ..sizevars import SimplifyIndexing
from ..utils import IndentedBuffer
from ..virtualized import V
from .simd import SIMDKernel
from .simd_kernel_features import DisableReduction, EnableReduction, SIMDKernelFeatures


@dataclass(frozen=True)
class HelionValue:
    name: str
    axes: tuple[str, ...]


@dataclass(frozen=True)
class HelionTensorArg:
    name: str
    source: str
    axes: tuple[str, ...]
    strides: tuple[sympy.Expr, ...]
    offset: sympy.Expr


class HelionKernel(DefaultHandler):
    """Emit a source-only preview of one native 2D matmul and its pointwise ops."""

    def __init__(
        self, features: SIMDKernelFeatures, tiling: dict[str, sympy.Expr]
    ) -> None:
        if tuple(tiling) != ("y", "x", "r0_") or not all(
            isinstance(size, sympy.Integer) for size in tiling.values()
        ):
            raise NotImplementedError("Helion preview requires a static 2D matmul")
        if features.op_counts().get("dot", 0) != 1:
            raise NotImplementedError("Helion preview requires a single matmul")
        if not config.triton.codegen_upcast_to_fp32:
            raise NotImplementedError("Helion preview requires FP32 compute upcasting")
        if any(
            V.graph.get_dtype(output.get_name()) not in (torch.float16, torch.bfloat16)
            for node in features.reduction_nodes()
            for output in node.get_outputs()
        ):
            raise NotImplementedError("Helion preview supports FP16/BF16 matmul")
        if features.get_mutations():
            raise NotImplementedError("Helion preview does not support mutations")
        self.features = features
        self.tiling = tiling
        self.symbols = dict(zip(tiling, sympy.symbols("helion_y helion_x helion_k")))
        self.tiles = {"y": "tile_y", "x": "tile_x", "r0_": "tile_k"}
        self.inputs: OrderedSet[str] = OrderedSet()
        nodes = tuple(features.scheduler_nodes())
        fused_names = OrderedSet(node.get_name() for node in nodes)
        self.produced = OrderedSet(
            output.get_name() for node in nodes for output in node.get_outputs()
        )
        self.outputs = OrderedSet(
            name
            for name in self.produced
            if not V.graph.scheduler.can_buffer_be_removed_through_fusion(
                name, fused_names
            )
        )
        self.local_buffers: dict[tuple[str, sympy.Expr], HelionValue] = {}
        self.arguments: dict[tuple[Any, ...], HelionTensorArg] = {}
        self.argument_counts: dict[str, int] = {}
        self.operand_axes: dict[torch.fx.Node, tuple[str, ...]] = {}
        self.reduction_body = IndentedBuffer()
        self.epilogue = IndentedBuffer()
        self.body = self.reduction_body
        self.counter = itertools.count()
        self.has_dot = False

    def find_operand_axes(self, graph: torch.fx.Graph) -> None:
        for node in graph.find_nodes(op="call_method", target="dot"):
            for operand, axes in zip(node.args[1:], [("y", "r0_"), ("r0_", "x")]):
                pending = [operand]
                visited = OrderedSet()
                while pending:
                    current = pending.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    if current.op == "call_method" and current.target == "load":
                        if (
                            current in self.operand_axes
                            and self.operand_axes[current] != axes
                        ):
                            raise NotImplementedError(
                                "Helion preview requires separate operand loads"
                            )
                        self.operand_axes[current] = axes
                    pending.extend(current.all_input_nodes)

    @staticmethod
    def compute_dtype(dtype: torch.dtype) -> torch.dtype:
        if config.triton.codegen_upcast_to_fp32 and dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            return torch.float32
        return dtype

    def emit(self, expression: str, axes: tuple[str, ...]) -> HelionValue:
        name = f"tmp{next(self.counter)}"
        self.body.writeline(f"{name} = {expression}")
        return HelionValue(name, axes)

    def align(self, value: HelionValue, axes: tuple[str, ...]) -> str:
        if not value.axes or value.axes == axes:
            return value.name
        if not OrderedSet(value.axes).issubset(axes):
            raise NotImplementedError("Helion preview cannot drop a tensor axis")
        order = tuple(axis for axis in axes if axis in value.axes)
        result = value.name
        if order != value.axes:
            permutation = [value.axes.index(axis) for axis in order]
            result += f".permute({', '.join(map(str, permutation))})"
        if len(axes) != len(value.axes):
            shape = [self.tiles[axis] if axis in value.axes else "1" for axis in axes]
            result += f".reshape([{', '.join(shape)}])"
        return result

    def tensor_arg(
        self, name: str, index: sympy.Expr, axes: tuple[str, ...]
    ) -> HelionTensorArg:
        strides = tuple(index.diff(self.symbols[axis]) for axis in axes)
        offset = index.subs(dict.fromkeys(self.symbols.values(), sympy.S.Zero))
        expected = offset + sum(
            stride * self.symbols[axis] for stride, axis in zip(strides, axes)
        )
        if (
            not all(isinstance(value, sympy.Integer) for value in (*strides, offset))
            or any(stride < 0 for stride in strides)
            or V.graph.sizevars.simplify(index - expected) != 0
        ):
            raise NotImplementedError(
                f"Non-affine Helion preview indexing: {name}[{index}]"
            )
        key = (name, axes, strides, offset)
        if key not in self.arguments:
            if name in self.outputs:
                prefix = "out"
            elif axes == ("y", "r0_"):
                prefix = "A"
            elif axes == ("r0_", "x"):
                prefix = "B"
            else:
                prefix = "in"
            number = self.argument_counts.get(prefix, 0)
            self.argument_counts[prefix] = number + 1
            alias = (
                prefix if number == 0 and prefix in ("A", "B") else f"{prefix}{number}"
            )
            self.arguments[key] = HelionTensorArg(alias, name, axes, strides, offset)
        return self.arguments[key]

    def load(self, name: str, index: sympy.Expr) -> HelionValue:
        if (name, index) in self.local_buffers:
            return self.local_buffers[name, index]
        if name in self.produced:
            raise NotImplementedError(
                "Helion preview requires elementwise buffer reuse"
            )
        axes = self.operand_axes.get(V.interpreter.current_node)
        if axes is None:
            axes = tuple(
                axis for axis, symbol in self.symbols.items() if index.has(symbol)
            )
        argument = self.tensor_arg(name, index, axes)
        self.inputs.add(name)
        indices = ", ".join(self.tiles[axis] for axis in axes)
        dtype = self.compute_dtype(V.graph.get_dtype(name))
        subscript = indices if axes else "()"
        return self.emit(f"{argument.name}[{subscript}].to({dtype})", axes)

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
        return self.emit(f"{value.name}.to({dtype})", value.axes)

    def dot(self, lhs: HelionValue, rhs: HelionValue) -> HelionValue:
        if self.has_dot:
            raise NotImplementedError("Helion preview supports one dot reduction")
        self.has_dot = True
        operands = []
        for value, axes in [(lhs, ("y", "r0_")), (rhs, ("r0_", "x"))]:
            operand = self.align(value, axes)
            if len(value.axes) != len(axes):
                shape = ", ".join(self.tiles[axis] for axis in axes)
                operand += f".expand([{shape}])"
            operands.append(operand)
        self.body.writeline(f"acc = hl.dot({operands[0]}, {operands[1]}, acc=acc)")
        return HelionValue("acc", ("y", "x"))

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
        if mode is not None:
            raise NotImplementedError("Helion preview does not support atomic stores")
        self.local_buffers[name, index] = value
        if name in self.outputs:
            axes = ("y", "x")
            argument = self.tensor_arg(name, index, axes)
            indices = ", ".join(self.tiles[axis] for axis in axes)
            stored = self.align(value, axes)
            dtype = V.graph.get_dtype(name)
            self.epilogue.writeline(
                f"hl.store({argument.name}, [{indices}], {stored}.to({dtype}))"
            )

    def store_reduction(self, name: str, index: sympy.Expr, value: HelionValue) -> None:
        self.store(name, index, value)

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        binary = {"add": "+", "sub": "-", "mul": "*", "truediv": "/"}
        if name in binary and not kwargs:
            lhs, rhs = args
            axes = tuple(dict.fromkeys((*lhs.axes, *rhs.axes)))
            expression = (
                f"{self.align(lhs, axes)} {binary[name]} {self.align(rhs, axes)}"
            )
            return self.emit(expression, axes)
        if name == "relu" and not kwargs:
            (value,) = args
            return self.emit(f"torch.relu({value.name})", value.axes)
        raise NotImplementedError(f"Unsupported Helion preview op: {name}")

    def codegen_kernel(self) -> str:
        def set_ranges(*ranges):
            if any(len(sizes) > 1 for sizes in ranges):
                raise NotImplementedError("Helion preview cannot split a logical axis")
            return [
                [symbol] if sizes else []
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
                if self.body is self.reduction_body and self.has_dot:
                    raise NotImplementedError(
                        "Helion preview supports one reduction phase"
                    )
                groups = list(self.tiling.values())
                if self.body is self.epilogue:
                    groups[-1] = sympy.S.One
                indices = SIMDKernel.map_kernel_groups_to_node_sizes(
                    groups, node.get_ranges(), set_ranges
                )
                self.find_operand_axes(node._body.root_block.graph)
                with V.set_ops_handler(SimplifyIndexing(self, var_ranges)):
                    node._body(*indices)

        code = IndentedBuffer()
        code.writelines(
            [
                "import torch",
                "import helion",
                "import helion.language as hl",
                "from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor",
                "",
            ]
        )
        code.writeline('@helion.kernel(static_shapes=True, dot_precision="ieee")')
        name = "helion_" + "_".join(
            n.get_name() for n in self.features.scheduler_nodes()
        )
        tensor_args = sorted(
            self.arguments.values(), key=lambda arg: arg.source in self.outputs
        )
        signature = ", ".join(f"{arg.name}: torch.Tensor" for arg in tensor_args)
        code.writeline(f"def {name}({signature}) -> None:")
        with code.indent():
            grid = f"[{self.tiling['y']}, {self.tiling['x']}]"
            code.writeline(f"for tile_y, tile_x in hl.tile({grid}):")
            with code.indent():
                code.writeline("acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)")
                code.writeline(f"for tile_k in hl.tile({self.tiling['r0_']}):")
                with code.indent():
                    code.splice(self.reduction_body)
                code.splice(self.epilogue)
        code.writeline("")
        signature = ", ".join(
            f"{arg}: torch.Tensor" for arg in (*self.inputs, *self.outputs)
        )
        code.writeline(f"def call_{name}({signature}) -> None:")
        with code.indent():
            for arg in tensor_args:
                size = tuple(self.tiling[axis] for axis in arg.axes)
                code.writeline(
                    f"{arg.name} = reinterpret_tensor({arg.source}, {size}, {arg.strides}, {arg.offset})"
                )
            arguments = ", ".join(arg.name for arg in tensor_args)
            code.writeline(f"{name}({arguments})")
        return code.getvalue()
