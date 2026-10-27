# Owner(s): ["module: inductor"]
from __future__ import annotations

import ast
import unittest
from unittest.mock import Mock, patch

import torch
from torch._inductor import config
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


class CodegenComplete(Exception):
    pass


@unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires CUDA scheduling and Triton")
@instantiate_parametrized_tests
class TestHelionCodegen(TestCase):
    hw_classification = HardwareClassification.CUDA

    def codegen(
        self, fn, shapes, *, strides=None, dtypes=None, helion_code=True, schedules=None
    ):
        fake_mode = FakeTensorMode(allow_fallback_kernels=False)
        with fake_mode:
            inputs = [
                torch.empty(shape, device="cuda", dtype=dtype)
                if strides is None
                else torch.empty_strided(shape, stride, device="cuda", dtype=dtype)
                for shape, stride, dtype in zip(
                    shapes,
                    strides or [None] * len(shapes),
                    dtypes or [torch.bfloat16] * len(shapes),
                )
            ]
            gm = make_fx(fn, tracing_mode="fake")(*inputs)
        debug = Mock()
        with (
            config.patch(
                {
                    "triton.native_matmul": True,
                    "trace.helion_code": helion_code,
                    "benchmark_fusion": False,
                    "max_autotune": False,
                    "max_autotune_gemm": False,
                    "graph_partition": False,
                }
            ),
            V.set_fake_mode(fake_mode),
            V.set_debug_handler(debug),
            patch.object(
                TritonScheduling, "create_kernel_choices", side_effect=CodegenComplete
            ),
        ):
            graph = GraphLowering(gm, example_inputs=inputs)
            with V.set_graph_handler(graph), V.set_extern_kernel_nodes([]):
                graph.run(*inputs)
                codegen_schedule = TritonScheduling.codegen_node_schedule

                def preview_native(scheduling, features):
                    if schedules is not None:
                        # Cache classification while native_matmul is patched on.
                        features.is_native_matmul()
                        schedules.append(features)
                    if features.is_native_matmul():
                        return codegen_schedule(scheduling, features)
                    for node in features.scheduler_nodes():
                        node.mark_run()

                with (
                    patch.object(
                        TritonScheduling, "codegen_node_schedule", preview_native
                    ),
                ):
                    try:
                        graph.codegen()
                    except CodegenComplete:
                        pass
        sources = [call.args[0] for call in debug.helion_code.call_args_list]
        for source in sources:
            compile(ast.parse(source), "<generated_helion>", "exec")
            self.assertNotIn(".view(-1)", source)
            self.assertNotIn(".reshape(-1)", source)
        return sources

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mm(self, dtype):
        (source,) = self.codegen(
            lambda a, b: a @ b, [(40, 72), (72, 136)], dtypes=[dtype, dtype]
        )
        self.assertIn("A[tile_y, tile_r]", source)
        self.assertIn("B[tile_r, tile_x]", source)
        self.assertIn("acc = hl.dot(tmp1, tmp3, acc=acc)", source)
        self.assertIn("out0[tile_y, tile_x] = acc.to(", source)
        self.assertIn("helion_op0(a_1, b_1, buf0)", source)
        self.assertNotIn("reinterpret_tensor", source)

    def test_bias_relu(self):
        schedules = []
        (source,) = self.codegen(
            lambda a, b, bias: torch.relu(a @ b + bias),
            [(40, 72), (72, 136), (136,)],
            schedules=schedules,
        )
        self.assertEqual(len(schedules), 1)
        self.assertGreater(len(tuple(schedules[0].scheduler_nodes())), 1)
        self.assertIn("in0[tile_x].to(torch.float32)", source)
        self.assertIn("tmp5 = acc + tmp4.reshape([1, tile_x])", source)
        self.assertIn("tmp6 = torch.relu(tmp5)", source)
        self.assertNotIn("hl.load(buf", source)

    @parametrize("axis", ["row", "column", "matrix"])
    def test_epilogue_broadcast(self, axis):
        shapes = {"row": (40,), "column": (136,), "matrix": (40, 136)}

        def fn(a, b, bias):
            return a @ b + (bias[:, None] if axis == "row" else bias)

        (source,) = self.codegen(fn, [(40, 72), (72, 136), shapes[axis]])
        coordinates = {"row": "tile_y", "column": "tile_x", "matrix": "tile_y, tile_x"}
        self.assertIn(f"in0[{coordinates[axis]}]", source)

    def test_nd_metadata_view(self):
        schedules = []
        (source,) = self.codegen(
            lambda a, b: a.reshape(33, 72) @ b,
            [(33, 9, 8), (72, 136)],
            schedules=schedules,
        )
        self.assertEqual(len(schedules), 1)
        self.assertIn("reinterpret_tensor(a_1, (33, 72), (72, 1), 0)", source)
        self.assertIn("A[tile_y, tile_r]", source)
        self.assertNotIn("//", source)
        self.assertNotIn("%", source)

    def test_nd_view_transpose_offset(self):
        (source,) = self.codegen(
            lambda a, bt: a[1:].reshape(33, 72) @ bt.T,
            [(34, 9, 8), (136, 72)],
        )
        self.assertIn("reinterpret_tensor(a_1, (33, 72), (72, 1), 72)", source)
        self.assertIn("reinterpret_tensor(bt_1, (72, 136), (1, 72), 0)", source)
        self.assertIn("B[tile_r, tile_x]", source)
        self.assertNotIn("//", source)
        self.assertNotIn("%", source)

    def test_caller_metadata_view(self):
        (source,) = self.codegen(
            lambda a, b: a[1:].reshape(33, 72) @ b,
            [(34, 9, 8), (72, 136)],
        )
        caller = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("call_")
        )
        kernel_name = caller.name.removeprefix("call_")
        kernel = Mock()
        namespace = {
            "torch": torch,
            "reinterpret_tensor": torch._C._dynamo.guards._reinterpret_tensor,
            kernel_name: kernel,
        }
        exec(
            compile(ast.Module(body=[caller], type_ignores=[]), "<caller>", "exec"),
            namespace,
        )
        a = torch.empty(34, 9, 8, dtype=torch.bfloat16)
        b = torch.empty(72, 136, dtype=torch.bfloat16)
        out = torch.empty(33, 136, dtype=torch.bfloat16)
        namespace[caller.name](a, b, out)
        lhs, rhs, result = kernel.call_args.args
        self.assertEqual(lhs.shape, (33, 72))
        self.assertEqual(lhs.stride(), (72, 1))
        self.assertEqual(lhs.storage_offset(), 72)
        self.assertEqual(lhs.untyped_storage()._cdata, a.untyped_storage()._cdata)
        self.assertIs(rhs, b)
        self.assertIs(result, out)

    def test_same_storage_two_operand_views(self):
        (source,) = self.codegen(lambda a: a @ a.T, [(40, 72)])
        self.assertIn("A[tile_y, tile_r]", source)
        self.assertIn("B[tile_r, tile_x]", source)
        self.assertIn("reinterpret_tensor(a_1, (72, 40), (1, 72), 0)", source)
        self.assertEqual(source.count("a_1: torch.Tensor"), 1)

    def test_strided_input(self):
        (source,) = self.codegen(
            lambda a, b: a @ b,
            [(40, 72), (72, 136)],
            strides=[(80, 1), (1, 80)],
        )
        self.assertIn("A[tile_y, tile_r]", source)
        self.assertIn("B[tile_r, tile_x]", source)
        self.assertNotIn("reinterpret_tensor", source)

    @parametrize("case", ["compute", "repeat", "gather", "reshape_copy"])
    def test_materialized_operand(self, case):
        if case == "compute":

            def fn(a, b):
                return (a.reshape(33, 72) + 1) @ (b - 2)

            shapes = [(33, 9, 8), (72, 136)]
            dtypes = None
        elif case == "repeat":

            def fn(a, b):
                return a.repeat(1, 9) @ b

            shapes = [(33, 8), (72, 136)]
            dtypes = None
        elif case == "gather":

            def fn(a, indices, b):
                return a[indices] @ b

            shapes = [(80, 72), (33,), (72, 136)]
            dtypes = [torch.bfloat16, torch.int64, torch.bfloat16]
        else:

            def fn(a, b):
                return a.permute(1, 0, 2).reshape(33, 64) @ b

            shapes = [(8, 33, 8), (64, 136)]
            dtypes = None
        schedules = []
        (source,) = self.codegen(fn, shapes, dtypes=dtypes, schedules=schedules)
        self.assertGreater(len(schedules), 1)
        native = schedules[-1]
        self.assertTrue(native.is_native_matmul())
        self.assertEqual(len(tuple(native.scheduler_nodes())), 1)
        self.assertIn("A[tile_y, tile_r]", source)
        self.assertIn("B[tile_r, tile_x]", source)
        self.assertIn("buf0: torch.Tensor", source)
        self.assertNotIn("a_1: torch.Tensor", source)
        self.assertNotIn("//", source)
        self.assertNotIn("%", source)
        self.assertNotIn("hl.full(", source)

    def test_producer_internal_fusion(self):
        schedules = []
        self.codegen(
            lambda a, b: ((a + 1) * 2) @ b,
            [(40, 72), (72, 136)],
            schedules=schedules,
        )
        self.assertEqual(len(schedules), 2)
        producer, native = schedules
        self.assertGreater(producer.op_counts().get("add", 0), 0)
        self.assertGreater(producer.op_counts().get("mul", 0), 0)
        self.assertEqual(len(tuple(native.scheduler_nodes())), 1)

    def test_triton_prologue_unchanged(self):
        schedules = []
        sources = self.codegen(
            lambda a, b: (a + 1) @ b,
            [(40, 72), (72, 136)],
            helion_code=False,
            schedules=schedules,
        )
        self.assertEqual(sources, [])
        self.assertEqual(len(schedules), 1)
        self.assertGreater(schedules[0].op_counts().get("add", 0), 0)

    def test_pointwise_not_selected(self):
        self.assertEqual(self.codegen(lambda a: a + 1, [(40, 72)]), [])

    def test_upcast_is_required(self):
        with config.patch("triton.codegen_upcast_to_fp32", False):
            with self.assertRaisesRegex(NotImplementedError, "FP32 compute upcasting"):
                self.codegen(lambda a, b: a @ b, [(40, 72), (72, 136)])


if __name__ == "__main__":
    run_tests()
