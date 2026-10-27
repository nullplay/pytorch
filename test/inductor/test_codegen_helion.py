# Owner(s): ["module: inductor"]
from __future__ import annotations

import ast
import unittest
from unittest.mock import Mock, patch

import torch
from torch._inductor import config
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.decomposition import select_decomp_table
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
        self,
        fn,
        shapes,
        *,
        strides=None,
        dtypes=None,
        helion_code=True,
        schedules=None,
        decompose=False,
        complete=False,
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
            gm = make_fx(
                fn,
                decomposition_table=select_decomp_table() if decompose else None,
                tracing_mode="fake",
            )(*inputs)
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
                        try:
                            return codegen_schedule(scheduling, features)
                        except CodegenComplete:
                            if not complete:
                                raise
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
        self.assertIn("acc = torch.addmm(acc, tmp1, tmp3)", source)
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

    @parametrize("epilogue", [False, True])
    def test_horizontal_mm(self, epilogue):
        def fn(a, b, c, d):
            first, second = a @ b, c @ d
            # Shared A makes this a horizontal candidate under the usual scheduler rules.
            return (
                torch.relu(first - second + first * second)
                if epilogue
                else (first, a @ d)
            )

        schedules = []
        (source,) = self.codegen(fn, [(40, 72), (72, 136)] * 2, schedules=schedules)
        self.assertEqual(len(schedules), 1)
        self.assertEqual(source.count(" = torch.addmm("), 2)
        self.assertIn("acc1 = hl.zeros(", source)
        self.assertIn("B1[tile_r, tile_x]", source)
        if epilogue:
            self.assertIn("A1[tile_y, tile_r]", source)
            self.assertIn("acc - acc1", source)
            self.assertIn("acc * acc1", source)
            self.assertIn("torch.relu(", source)

    def test_horizontal_then_epilogue_chain(self):
        (source,) = self.codegen(
            lambda a, b, c, d, e, f: torch.relu(a @ b + c @ d + e @ f),
            [(40, 72), (72, 136)] * 3,
        )
        self.assertEqual(source.count(" = torch.addmm("), 3)
        self.assertIn("acc2 = hl.zeros(", source)
        self.assertIn("torch.relu(", source)

    @parametrize("horizontal", [False, True])
    def test_bmm_epilogue(self, horizontal):
        def fn(a, b, c, d, bias):
            first = torch.bmm(a, b)
            result = first - torch.bmm(c, d) if horizontal else first
            return torch.relu(result) + first * first + bias

        schedules = []
        (source,) = self.codegen(
            fn,
            [(3, 40, 72), (3, 72, 136)] * 2 + [(136,)],
            schedules=schedules,
        )
        self.assertEqual(len(schedules), 1)
        self.assertIn("for tile_z, tile_y, tile_x in hl.tile([3, 40, 136]):", source)
        self.assertIn("A[tile_z, tile_y, tile_r]", source)
        self.assertIn("B[tile_z, tile_r, tile_x]", source)
        self.assertIn("out0[tile_z, tile_y, tile_x]", source)
        self.assertEqual(source.count(" = torch.baddbmm("), 2 if horizontal else 1)

    def test_bmm_operand_view(self):
        (source,) = self.codegen(
            lambda a, bt: torch.bmm(a.reshape(3, 40, 72), bt.permute(0, 2, 1)),
            [(3, 40, 9, 8), (3, 136, 72)],
        )
        self.assertIn("reinterpret_tensor(a_1, (3, 40, 72), (2880, 72, 1), 0)", source)
        self.assertIn(
            "reinterpret_tensor(bt_1, (3, 72, 136), (9792, 1, 72), 0)", source
        )
        self.assertNotIn("//", source)
        self.assertNotIn("%", source)

    def test_fp32(self):
        (source,) = self.codegen(
            lambda a, b: a @ b,
            [(40, 72), (72, 136)],
            dtypes=[torch.float32, torch.float32],
        )
        self.assertIn("acc = torch.addmm(acc, tmp0, tmp1)", source)

    @parametrize("batched", [False, True])
    def test_atomic_add_epilogue(self, batched):
        def fn(a, b, indices, out):
            value = torch.bmm(a, b) if batched else a @ b
            out.index_add_(0, indices, value)
            return out

        shapes = (
            [(3, 40, 72), (3, 72, 136), (3,), (7, 40, 136)]
            if batched
            else [(40, 72), (72, 136), (40,), (80, 136)]
        )
        (source,) = self.codegen(
            fn,
            shapes,
            dtypes=[torch.float32, torch.float32, torch.int64, torch.float32],
            decompose=True,
            complete=True,
        )
        self.assertIn("hl.atomic_add(", source)
        self.assertIn("out_1", source)
        self.assertIn("acc", source)

    def test_gather_mm_broadcast_epilogue(self):
        # The normal memory heuristic can reject this fusion after input realization.
        # Bypass only that heuristic to check the legal fused schedule's codegen.
        with patch(
            "torch._inductor.scheduler.Scheduler.can_fusion_increase_peak_memory",
            return_value=False,
        ):
            (source,) = self.codegen(
                lambda x, y, z, w: x[z] @ y + w + 3,
                [(128, 128)] * 4,
                dtypes=[torch.float32, torch.float32, torch.int64, torch.float32],
                decompose=True,
                complete=True,
            )
        self.assertIn("torch.addmm(", source)
        self.assertIn("in0[", source)
        self.assertIn("3.0", source)
        self.assertIn("% 128", source)

    def test_scalar_tensor_epilogue(self):
        (source,) = self.codegen(
            lambda a, b, bias: a @ b + bias,
            [(40, 72), (72, 136), ()],
        )
        self.assertIn("in0[()]", source)

    def test_padded_scatter_rejected(self):
        def fn(a, b, indices, out):
            out.index_add_(0, indices, a @ b)
            return out

        with self.assertRaisesRegex(
            NotImplementedError, "Cannot recover Helion tensor indices"
        ):
            self.codegen(
                fn,
                [(40, 72), (72, 136), (40,), (80, 136)],
                strides=[(72, 1), (136, 1), (1,), (144, 1)],
                dtypes=[torch.float32, torch.float32, torch.int64, torch.float32],
                decompose=True,
                complete=True,
            )

    def test_upcast_is_required(self):
        with config.patch("triton.codegen_upcast_to_fp32", False):
            with self.assertRaisesRegex(NotImplementedError, "FP32 compute upcasting"):
                self.codegen(lambda a, b: a @ b, [(40, 72), (72, 136)])


if __name__ == "__main__":
    run_tests()
