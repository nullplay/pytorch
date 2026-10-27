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

    def codegen(self, fn, shapes, *, strides=None, dtypes=None):
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
                    "trace.helion_code": True,
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
                with self.assertRaises(CodegenComplete):
                    graph.codegen()
        sources = [call.args[0] for call in debug.helion_code.call_args_list]
        for source in sources:
            compile(ast.parse(source), "<generated_helion>", "exec")
        return sources

    @parametrize("shape", [(128, 128, 128), (40, 72, 136)])
    def test_mm(self, shape):
        m, k, n = shape
        (source,) = self.codegen(lambda a, b: a @ b, [(m, k), (k, n)])
        self.assertIn(f"for tile_y, tile_x in hl.tile([{m}, {n}]):", source)
        self.assertIn(f"for tile_k in hl.tile({k}):", source)
        self.assertIn("A[tile_y, tile_k]", source)
        self.assertIn("B[tile_k, tile_x]", source)
        self.assertIn("acc = hl.dot(tmp1, tmp3, acc=acc)", source)
        self.assertIn(
            "hl.store(out0, [tile_y, tile_x], acc.to(torch.bfloat16))", source
        )
        self.assertIn(f"A = reinterpret_tensor(a_1, ({m}, {k}), ({k}, 1), 0)", source)
        self.assertIn(f"B = reinterpret_tensor(b_1, ({k}, {n}), ({n}, 1), 0)", source)

    def test_bias_relu(self):
        (source,) = self.codegen(
            lambda a, b, bias: torch.relu(a @ b + bias),
            [(128, 128), (128, 128), (128,)],
        )
        self.assertIn("in0[tile_x].to(torch.float32)", source)
        self.assertIn("tmp5 = acc + tmp4.reshape([1, tile_x])", source)
        self.assertIn("tmp6 = torch.relu(tmp5)", source)
        self.assertNotIn("hl.load(buf0", source)
        self.assertNotIn("hl.store(buf0", source)
        self.assertNotIn("acc.to(torch.bfloat16)", source)

    def test_prologue_tail(self):
        (source,) = self.codegen(lambda a, b: (a + 1) @ (b - 2), [(40, 72), (72, 136)])
        self.assertExpectedInline(
            source,
            """\
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision="ieee")
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([40, 136]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_k in hl.tile(72):
            tmp0 = A[tile_y, tile_k].to(torch.float32)
            tmp1 = hl.full([], 1.0, dtype=torch.float32)
            tmp2 = tmp0 + tmp1
            tmp3 = tmp2.to(torch.bfloat16)
            tmp4 = B[tile_k, tile_x].to(torch.float32)
            tmp5 = hl.full([], 2.0, dtype=torch.float32)
            tmp6 = tmp4 - tmp5
            tmp7 = tmp6.to(torch.bfloat16)
            acc = hl.dot(tmp3, tmp7, acc=acc)
        hl.store(out0, [tile_y, tile_x], acc.to(torch.bfloat16))

def call_helion_op0(a_1: torch.Tensor, b_1: torch.Tensor, buf0: torch.Tensor) -> None:
    A = reinterpret_tensor(a_1, (40, 72), (72, 1), 0)
    B = reinterpret_tensor(b_1, (72, 136), (136, 1), 0)
    out0 = reinterpret_tensor(buf0, (40, 136), (136, 1), 0)
    helion_op0(A, B, out0)
""",
        )

    def test_strided_transpose(self):
        (source,) = self.codegen(
            lambda a, bt: a @ bt.T,
            [(63, 80), (96, 80)],
            strides=[(1, 63), (80, 1)],
        )
        self.assertIn("A[tile_y, tile_k]", source)
        self.assertIn("B[tile_k, tile_x]", source)
        self.assertIn("A = reinterpret_tensor(a_1, (63, 80), (1, 63), 0)", source)
        self.assertIn("B = reinterpret_tensor(bt_1, (80, 96), (1, 80), 0)", source)
        self.assertIn("acc = hl.dot(tmp1, tmp3, acc=acc)", source)

    def test_returned_intermediate(self):
        def fn(a, b, bias):
            mm = a @ b
            return mm, torch.relu(mm + bias)

        (source,) = self.codegen(fn, [(40, 72), (72, 136), (136,)])
        self.assertIn(
            "hl.store(out0, [tile_y, tile_x], acc.to(torch.bfloat16))", source
        )
        self.assertIn("tmp5 = acc + tmp4.reshape([1, tile_x])", source)
        self.assertNotIn("hl.load(buf0", source)
        self.assertIn("buf0: torch.Tensor, buf1: torch.Tensor", source)

    def test_pointwise_not_selected(self):
        self.assertEqual(self.codegen(lambda a: a + 1, [(40, 72)]), [])

    def test_offset_view(self):
        (source,) = self.codegen(lambda a, b: a[1:] @ b, [(41, 72), (72, 136)])
        self.assertIn("A[tile_y, tile_k]", source)
        self.assertIn("A = reinterpret_tensor(a_1, (40, 72), (72, 1), 72)", source)

    def test_broadcast_view(self):
        (source,) = self.codegen(lambda a, b: a.expand(40, 72) @ b, [(72,), (72, 136)])
        self.assertIn("A[tile_y, tile_k]", source)
        self.assertIn("A = reinterpret_tensor(a_1, (40, 72), (0, 1), 0)", source)

    def test_gather_not_supported(self):
        with self.assertRaisesRegex(
            NotImplementedError, "Unsupported Helion preview op"
        ):
            self.codegen(
                lambda pool, indices, b: pool[indices] @ b,
                [(80, 72), (63,), (72, 96)],
                dtypes=[torch.bfloat16, torch.int64, torch.bfloat16],
            )

    def test_upcast_is_required(self):
        with config.patch("triton.codegen_upcast_to_fp32", False):
            with self.assertRaisesRegex(NotImplementedError, "FP32 compute upcasting"):
                self.codegen(lambda a, b: a @ b, [(40, 72), (72, 136)])

    def test_integer_not_supported(self):
        with self.assertRaisesRegex(NotImplementedError, "FP16/BF16 matmul"):
            self.codegen(
                lambda a, b: a @ b,
                [(40, 72), (72, 136)],
                dtypes=[torch.int8, torch.int8],
            )

    def test_nonfinite_constant(self):
        (source,) = self.codegen(
            lambda a, b: (a + float("inf")) @ b, [(40, 72), (72, 136)]
        )
        self.assertIn("hl.full([], float('inf'), dtype=torch.float32)", source)

    def test_bmm_not_supported(self):
        with self.assertRaisesRegex(NotImplementedError, "static 2D matmul"):
            self.codegen(lambda a, b: a @ b, [(3, 40, 72), (3, 72, 136)])

    def test_two_mm_not_supported(self):
        with self.assertRaisesRegex(NotImplementedError, "single matmul"):
            self.codegen(
                lambda a, b, c, d: a @ b + c @ d,
                [(64, 128), (128, 96), (64, 128), (128, 96)],
            )


if __name__ == "__main__":
    run_tests()
