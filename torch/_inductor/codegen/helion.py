from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, cast

import sympy

import torch
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..ir import IRNode, MutableBox, ReinterpretView
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
    depending_indices: tuple[str, ...]


@dataclass(frozen=True)
class HelionTensorArg:
    name: str
    operand: IRNode

    @property
    def source(self) -> str:
        return self.operand.get_name()


class HelionKernel(DefaultHandler):
    """Emit a source-only preview of one 2D matmul and pointwise epilogues."""

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
        if features.get_mutations():
            raise NotImplementedError("Helion preview does not support mutations")
        nodes = tuple(features.scheduler_nodes())
        matmuls = [
            V.graph.helion_matmul_inputs[name]
            for node in nodes
            for name in node.get_buffer_names()
            if name in V.graph.helion_matmul_inputs
        ]
        if len(matmuls) != 1:
            raise NotImplementedError("Helion preview requires lowered MM operands")
        self.matmul_inputs = matmuls[0]
        if any(
            operand.get_dtype() not in (torch.float16, torch.bfloat16)
            for operand in self.matmul_inputs
        ):
            raise NotImplementedError("Helion preview supports FP16/BF16 matmul")
        self.features = features
        self.tiling = tiling
        self.symbols = dict(
            zip(tiling, sympy.symbols("helion_y helion_x helion_r", integer=True))
        )
        self.tiles = {"y": "tile_y", "x": "tile_x", "r0_": "tile_r"}
        self.inputs: OrderedSet[str] = OrderedSet()
        fused_names = OrderedSet(node.get_name() for node in nodes)
        self.outputs = OrderedSet(
            output.get_name()
            for node in nodes
            for output in node.get_outputs()
            if not V.graph.scheduler.can_buffer_be_removed_through_fusion(
                output.get_name(), fused_names
            )
        )
        self.arguments: dict[tuple[Any, ...], HelionTensorArg] = {}
        self.store_forwarding: dict[tuple[str, sympy.Expr], HelionValue] = {}
        self.reduction_body = IndentedBuffer()
        self.epilogue = IndentedBuffer()
        self.body = self.reduction_body
        self.counter = itertools.count()
        self.has_dot = False

    @staticmethod
    def compute_dtype(dtype: torch.dtype) -> torch.dtype:
        if config.triton.codegen_upcast_to_fp32 and dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            return torch.float32
        return dtype

    def emit(self, expression: str, depending_indices: tuple[str, ...]) -> HelionValue:
        name = f"tmp{next(self.counter)}"
        self.body.writeline(f"{name} = {expression}")
        return HelionValue(name, depending_indices)

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
            result += ".T"
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

    def epilogue_indices(
        self, operand: IRNode, index: sympy.Expr
    ) -> tuple[list[str], tuple[str, ...]]:
        layout = operand.get_layout()
        if len(layout.size) not in (1, 2):
            raise NotImplementedError("Helion epilogues require 1D/2D tensor inputs")
        # For shape (M, N), stride (N, 1), match N*y + x to [tile_y, tile_x].
        # A bias with shape (N,) and index x becomes bias[tile_x].
        candidates = [
            [
                axis
                for axis in ("y", "x")
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
                    [self.tiles[axis] if axis is not None else "0" for axis in axes],
                    active,
                )
        raise NotImplementedError(f"Unsupported Helion epilogue indexing: {index}")

    def load(self, name: str, index: sympy.Expr) -> HelionValue:
        index = sympy.sympify(index)
        if (name, index) in self.store_forwarding:
            return self.store_forwarding[name, index]
        if self.body is self.reduction_body:
            for operand, axes, alias in (
                (self.matmul_inputs[0], ("y", "r0_"), "A"),
                (self.matmul_inputs[1], ("r0_", "x"), "B"),
            ):
                if operand.get_name() != name:
                    continue
                expected = operand.make_indexer()([self.symbols[axis] for axis in axes])
                if V.graph.sizevars.statically_known_equals(index, expected):
                    argument = self.tensor_arg(operand, alias)
                    self.inputs.add(name)
                    coordinates = ", ".join(self.tiles[axis] for axis in axes)
                    dtype = self.compute_dtype(operand.get_dtype())
                    return self.emit(
                        f"{argument.name}[{coordinates}].to({dtype})", axes
                    )
            raise NotImplementedError("Helion preview does not support fused prologues")
        operand = V.graph.get_buffer(name)
        coordinates, axes = self.epilogue_indices(operand, index)
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
        if self.has_dot:
            raise NotImplementedError("Helion preview supports one dot reduction")
        self.has_dot = True
        operands = []
        for value, depending_indices in [(lhs, ("y", "r0_")), (rhs, ("r0_", "x"))]:
            operand = self.align(value, depending_indices)
            if len(value.depending_indices) != len(depending_indices):
                shape = ", ".join(self.tiles[axis] for axis in depending_indices)
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
        self.store_forwarding[name, sympy.sympify(index)] = value
        if name in self.outputs:
            operand = V.graph.get_buffer(name)
            coordinates, axes = self.epilogue_indices(operand, index)
            argument = self.tensor_arg(operand)
            stored = self.align(value, axes)
            self.epilogue.writeline(
                f"{argument.name}[{', '.join(coordinates)}] = "
                f"{stored}.to({operand.get_dtype()})"
            )

    def store_reduction(self, name: str, index: sympy.Expr, value: HelionValue) -> None:
        self.store(name, index, value)

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        binary = {"add": "+", "sub": "-", "mul": "*", "truediv": "/"}
        if name in binary and not kwargs:
            lhs, rhs = args
            depending_indices = tuple(
                dict.fromkeys((*lhs.depending_indices, *rhs.depending_indices))
            )
            left = self.align(lhs, depending_indices)
            right = self.align(rhs, depending_indices)
            expression = f"{left} {binary[name]} {right}"
            return self.emit(expression, depending_indices)
        if name == "relu" and not kwargs:
            (value,) = args
            return self.emit(f"torch.relu({value.name})", value.depending_indices)
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
            code.writeline(
                f"for tile_y, tile_x in hl.tile([{self.tiling['y']}, {self.tiling['x']}]):"
            )
            with code.indent():
                code.writeline("acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)")
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
