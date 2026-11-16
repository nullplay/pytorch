# Owner(s): ["module: inductor"]


import itertools
from collections.abc import Callable

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config as inductor_config, metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import (
    fresh_inductor_cache,
    run_and_get_code,
    run_and_get_triton_code,
)
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    largeTensorTest,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


aten = torch.ops.aten


@inductor_config.patch({"triton.native_matmul": True})
class TestTritonDotReduction(TestCase):
    def _check_equal(
        self, f: Callable, example_inputs: tuple[torch.Tensor], tol: float = 1e-4
    ):
        compiled = torch.compile(f)
        actual = compiled(*example_inputs)
        expect = f(*example_inputs)
        self.assertTrue(same(expect, actual, tol=tol))

    def _check_code(
        self,
        f: Callable,
        example_inputs: tuple[torch.Tensor],
        kernel_count: int,
        dot_count: int,
    ):
        f = torch.compile(f)
        code = run_and_get_triton_code(f, *example_inputs)
        FileCheck().check_regex(r"triton.*mm.*\.run\(").run(code)

        FileCheck().check_count("@triton.jit", kernel_count, exactly=True).check_count(
            "tl.dot", dot_count, exactly=True
        ).run(code)

    def test_matmul(self):
        def f(x, y):
            z = x @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    def test_mm_1d_expand(self):
        def f(x, y, M, K):
            z = x[:, None].expand(M, K) @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M,), (1,), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, M, K))
        self._check_code(f, (x, y, M, K), 1, 1)

    def test_mm_2_expand(self):
        def f(x, y, M, K):
            z = x[:, None].expand(M, K) @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((1,), (0,), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, M, K))
        self._check_code(f, (x, y, M, K), 1, 1)

    def test_matmul_fp16(self):
        def f(x, y):
            z = x @ y.to(x.dtype)
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), dtype=torch.float16, device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), dtype=torch.float32, device=GPU_TYPE)

        # _check_equal calls torch._dynamo.utils.same with kwarg tol=1e-4.
        # For fp16 dtype, torch.allclose() defaults to atol=1e-3 rtol=1e-5,
        # but same() uses the single value to assign both, resulting in
        # Accuracy failed: allclose not within tol=0.0001.
        self._check_equal(f, (x, y), tol=1e-3)
        self._check_code(f, (x, y), 1, 1)

    def test_reduction_mask_zeroout(self):
        def f(x, y):
            return (x + 1) @ (y - 2)

        M, K, N = 62, 62, 62
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    def test_3mm_add(self):
        def f(x, y, z, w, r, t):
            return x @ y + z @ w + r @ t

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)
        w = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        z = rand_strided((K, N), (N, 1), device=GPU_TYPE)
        r = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        t = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, z, w, r, t))
        self._check_code(f, (x, y, z, w, r, t), 1, 3)

    def test_mm_complex(self):
        def f(x, y, z, w):
            return x[z] @ y + w + 3

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        z = torch.randint(M, (M, K), dtype=torch.long, device=GPU_TYPE)
        w = rand_strided((M, N), (N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, z, w))
        self._check_code(f, (x, y, z, w), 1, 1)

    def test_batchmatmul(self):
        def f(x, y):
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 256, 128, 128, 128
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    def test_bmm_vertical_fusion(self):
        def f(x, y):
            z = torch.bmm(x, y)
            w = torch.nn.functional.relu(z)
            v = w + z * z
            return v

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    def test_bmm_horizontal_fusion(self):
        def f(x, y, z, w):
            bmm1 = torch.bmm(x, y)
            bmm2 = torch.bmm(z, w)
            return bmm1 - bmm2 + bmm1 * bmm2

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)
        z = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        w = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        self._check_equal(f, (x, y, z, w))
        self._check_code(f, (x, y, z, w), 1, 2)

    def test_bmm_fusion_complex1(self):
        # Out[O[b,i],j] += Input[I[b,i],r] * Weight[W[b],r,j] * Val[b,i]
        def f(inp, weight, val, I_idx, W_idx, O_idx, out):
            x = inp[I_idx]  # [B,I,R]
            y = weight[W_idx]  # [B,R,J]
            t = torch.einsum("bir,brj,bi->bij", x, y, val)
            out.index_add_(0, O_idx.reshape(-1), t.reshape(-1, J))
            return out

        B, I, R, J = 32, 16, 128, 16
        N_in = 128
        N_w = 64
        N_out = 128

        inp = rand_strided((N_in, R), (R, 1), device=GPU_TYPE)
        weight = rand_strided((N_w, R, J), (R * J, J, 1), device=GPU_TYPE)
        val = rand_strided((B, I), (I, 1), device=GPU_TYPE)

        I_idx = torch.randint(0, N_in, (B, I), device=GPU_TYPE, dtype=torch.int64)
        W_idx = torch.randint(0, N_w, (B,), device=GPU_TYPE, dtype=torch.int64)
        O_idx = torch.randint(0, N_out, (B, I), device=GPU_TYPE, dtype=torch.int64)

        out = torch.zeros((N_out, J), device=GPU_TYPE)

        self._check_equal(f, (inp, weight, val, I_idx, W_idx, O_idx, out))
        self._check_code(f, (inp, weight, val, I_idx, W_idx, O_idx, out), 1, 1)

    def test_bmm_fusion_complex2(self):
        # out[Ai[g],m,n] += Av[g,m,k,p] * B[Ak[g,p],k,n]
        def f(Av, B, Ai, Ak, out):
            Bg = B[Ak]  # [G,P,K,N]
            Cg = torch.einsum("gmkp,gpkn->gmn", Av, Bg)  # [G,M,N]
            out.index_add_(0, Ai, Cg)
            return out

        G, M, K, P, N = 128, 16, 64, 8, 16
        NB = 128
        NA = 128

        Av = rand_strided((G, M, K, P), (M * K * P, K * P, P, 1), device=GPU_TYPE)
        B = rand_strided((NB, K, N), (K * N, N, 1), device=GPU_TYPE)

        Ai = torch.randint(0, NA, (G,), device=GPU_TYPE, dtype=torch.int64)
        Ak = torch.randint(0, NB, (G, P), device=GPU_TYPE, dtype=torch.int64)

        out = torch.zeros((NA, M, N), device=GPU_TYPE)

        self._check_equal(f, (Av, B, Ai, Ak, out))
        self._check_code(f, (Av, B, Ai, Ak, out), 1, 1)

    def test_bmm_large_batch_reversed_pid(self):
        def f(x, y):
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 65537, 16, 16, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        f = torch.compile(f)
        code = run_and_get_triton_code(f, x, y)

        FileCheck().check("z0_offset = tl.program_id(0)").check(
            "yoffset = tl.program_id(1)"
        ).check("xoffset = tl.program_id(2)").run(code)

    def test_bmm_no_z_broadcast(self):
        def f(x, y):
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K * N, N, 1), device=GPU_TYPE)

        f = torch.compile(f)
        code = run_and_get_triton_code(f, x, y)

        check = FileCheck().check_not("tl.arange(0, Z0_BLOCK)[:, None, None, None]")
        check.check("tl.arange(0, Z0_BLOCK)").check(
            "tl.arange(0, YBLOCK)[None, :, None, None]"
        ).check("tl.arange(0, XBLOCK)[None, None, :, None]").check(
            "tl.arange(0, R0_BLOCK)[None, None, None, :]"
        ).run(code)


@inductor_config.patch({"triton.native_matmul": True})
@instantiate_parametrized_tests
class TestNativeMatmulRowReduction(TestCase):
    """[M, N] -> [M] reductions of a native matmul fuse into the matmul kernel."""

    def check(self, f, args, *, fused, tol=5e-2):
        metrics.reset()
        torch._dynamo.reset()
        actual, code = run_and_get_code(torch.compile(f), *args)
        self.assertEqual(actual, f(*args), atol=tol, rtol=tol)
        self.assertEqual(metrics.generated_kernel_count == 1, fused)
        return code[0]

    @parametrize(
        "shape",
        (
            (256, 128, 16),
            (256, 2048, 64),
            (256, 1024, 128),
            (256, 128, 62),
            (512, 4096, 64),
            (128, 64, 64),
        ),
    )
    def test_output_reduction(self, shape):
        def f(x, w1, bias1, w2, bias2):
            hidden = torch.relu(x @ w1 + bias1)
            return (hidden * w2).sum(dim=-1) + bias2

        m, k, n = shape
        args = (
            torch.randn(m, k, dtype=torch.float16),
            torch.randn(k, n, dtype=torch.float16),
            torch.randn(n, dtype=torch.float16),
            torch.randn(n, dtype=torch.float16),
            torch.randn(1, dtype=torch.float16),
        )
        self.check(f, args, fused=k <= 512, tol=5e-1)

    def test_reduced_epilogue(self):
        def f(x, weight, scale):
            return torch.sqrt((x @ weight).sum(-1).abs()) * scale

        args = (
            torch.randn(100, 64, dtype=torch.float16),
            torch.randn(64, 100, dtype=torch.float16),
            torch.randn(100, dtype=torch.float16),
        )
        code = self.check(f, args, fused=True)
        # The row stage is rank 2, so [M] stores are not repeated across a row.
        FileCheck().check("tl.sum(").check("tl.store(").check_same(
            "local_xindex_mask)"
        ).run(code)

    def test_matmul_output_precision(self):
        def f(x, weight):
            return (x @ weight).sum(-1)

        x = torch.ones((16, 16), dtype=torch.float16)
        weight = torch.empty((16, 8), dtype=torch.float16)
        weight[:, 0::2] = 5000
        weight[:, 1::2] = -5000
        code = self.check(f, (x, weight), fused=True, tol=0)
        FileCheck().check("Original ATen: [aten.mm, aten.sum]").check("tl.dot").check(
            ".to(tl.float16)"
        ).check("tl.sum").run(code)

    def test_row_reduction_output_precision(self):
        def f(x, weight):
            reduced = (x @ weight).sum(-1)
            return (reduced - 4.3984375) * 1000

        x = torch.zeros((16, 16), dtype=torch.float16)
        x[:, 0] = 1
        weight = torch.zeros((16, 8), dtype=torch.float16)
        weight[0, ::2] = 1
        weight[0, 1::2] = 0.1
        code = self.check(f, (x, weight), fused=True, tol=0)
        FileCheck().check("tl.dot").check("tl.sum").check(".to(tl.float16)").run(code)

    def test_indirect_index_from_row_reduction_not_fused(self):
        def f(x, weight, table):
            index = (x @ weight).amax(-1).clamp(0, 15).long()
            return table[index] * 2

        args = (
            torch.rand(100, 64, dtype=torch.float16),
            torch.rand(64, 48, dtype=torch.float16) * 0.01,
            torch.randn(16),
        )
        self.check(f, args, fused=False, tol=0)

    @parametrize("reduction_type", ("prod", "any"))
    def test_additional_reduction_types(self, reduction_type):
        def f(x, weight):
            output = x @ weight
            return output.prod(-1) if reduction_type == "prod" else output.any(-1)

        args = (
            torch.eye(16, dtype=torch.float16).repeat(2, 1),
            torch.full((16, 8), 0.5, dtype=torch.float16),
        )
        self.check(f, args, fused=True, tol=1e-3)

    @parametrize("reduction_type", ("amax", "amin"))
    def test_padded_lanes(self, reduction_type):
        def f(x, weight):
            output = x @ weight
            if reduction_type == "amax":
                return output.amax(dim=-1)
            return output.amin(dim=-1)

        sign = -1 if reduction_type == "amax" else 1
        args = (
            torch.ones(16, 16, dtype=torch.float16),
            torch.full((16, 8), sign, dtype=torch.float16),
        )
        code = self.check(f, args, fused=True, tol=0)
        FileCheck().check("tl.where(local_r0_index_mask").run(code)

    def test_atomic_add(self):
        def f(x, weight, index, out):
            return out.index_add_(0, index, (x @ weight).sum(dim=-1))

        torch.manual_seed(6928)
        args = (
            torch.randn(16, 16, dtype=torch.float16),
            torch.randn(16, 8, dtype=torch.float16),
            torch.zeros(16, dtype=torch.int64),
        )
        out = torch.zeros(1, dtype=torch.float16)
        expected = f(*args, out.clone())
        with (
            fresh_inductor_cache(),
            inductor_config.patch(epilogue_fusion_with_atomic_add=True),
        ):
            actual, code = run_and_get_code(torch.compile(f), *args, out.clone())
        self.assertEqual(actual, expected, atol=5e-2, rtol=5e-2)
        FileCheck().check_count("@triton.jit", 1, exactly=True).check(
            "tl.atomic_add"
        ).run(code[0])
        FileCheck().check_not("'mutated_arg_names': []").run(code[0])

    @largeTensorTest("3GB", device=GPU_TYPE, inductor=True)
    def test_index_dtype(self):
        def f(x, weight, scale):
            return ((x @ weight) * scale).sum(dim=-1)

        # Only the row stage reads scale, and its storage spans > 2**31
        # elements, so the fused kernel must use 64-bit indexing.
        stride = 2**27 + 2**24
        scale_storage = torch.ones(15 * stride + 1, dtype=torch.uint8)
        args = (
            torch.randn(16, 16, dtype=torch.float16),
            torch.randn(16, 16, dtype=torch.float16),
            scale_storage[::stride],
        )
        code = self.check(f, args, fused=True)
        if not TEST_WITH_ROCM:
            FileCheck().check(".to(tl.int64)").run(code)

    def test_reused_prologue(self):
        # K == N, so the [M, K] prologue and [M, N] output share a numel.
        def f(x, weight):
            prologue = x + 1
            output = prologue @ weight
            return (output + prologue).sum(dim=-1)

        args = (
            torch.randn(64, 64, dtype=torch.float16),
            torch.randn(64, 64, dtype=torch.float16),
        )
        self.check(f, args, fused=True)

    def test_column_reduction_not_fused(self):
        def f(x, weight):
            return (x @ weight).sum(dim=0)

        args = (
            torch.randn(16, 32, dtype=torch.float16),
            torch.randn(32, 16, dtype=torch.float16),
        )
        self.check(f, args, fused=False)

    def test_mismatched_row_reduction_not_fused(self):
        def f(x, weight, other):
            return (x @ weight).sum(-1) + other.sum(-1)

        args = (
            torch.randn(16, 16, dtype=torch.float16),
            torch.randn(16, 16, dtype=torch.float16),
            torch.randn(16, 32, dtype=torch.float16),
        )
        self.check(f, args, fused=False)

    def test_transposed_pointwise_not_fused(self):
        def f(x, weight):
            output = torch.ops._inductor_test.realize((x @ weight).T + 1)
            return output.sum(dim=-1)

        args = (
            torch.randn(64, 32, dtype=torch.float16),
            torch.randn(32, 64, dtype=torch.float16),
        )
        self.check(f, args, fused=False)

    @parametrize("case", ("flattened", "reshaped"))
    def test_partial_row_reduction_not_fused(self, case):
        def f(x, weight):
            output = x @ weight
            if case == "flattened":
                return output.reshape(1, -1).expand(64, -1).sum(-1)
            return output.reshape(32, 1, 8).expand(32, 32, 8).amax(-1)

        m, n = (4, 8) if case == "flattened" else (16, 16)
        args = (
            torch.randn(m, 64, dtype=torch.float16),
            torch.randn(64, n, dtype=torch.float16),
        )
        self.check(f, args, fused=False)

    @parametrize("op", ("sort", "cumsum"))
    def test_sort_scan_not_fused(self, op):
        def f(x, weight):
            output = x @ weight
            return output.sort(-1).values if op == "sort" else output.cumsum(-1)

        args = (torch.randn(64, 32), torch.randn(32, 16))
        self.check(f, args, fused=False, tol=1e-4)

    def test_broadcast_consumer_not_fused(self):
        def f(x, weight):
            hidden = x @ weight
            return hidden - hidden.sum(-1, keepdim=True)

        args = (
            torch.randn(256, 128, dtype=torch.float16),
            torch.randn(128, 16, dtype=torch.float16),
        )
        self.check(f, args, fused=False)


@inductor_config.patch({"triton.native_matmul": True})
@instantiate_parametrized_tests
class TestNativeMatmulDotStage(TestCase):
    """A native matmul that contracts the N columns of another native matmul's
    complete rows (a back-to-back GEMM) fuses into that matmul's kernel."""

    def check(self, f, args, *, fused, tol=5e-2):
        metrics.reset()
        torch._dynamo.reset()
        actual, code = run_and_get_code(torch.compile(f), *args)
        self.assertEqual(actual, f(*args), atol=tol, rtol=tol)
        self.assertEqual(metrics.generated_kernel_count == 1, fused)
        return code[0]

    @parametrize("shape", ((8, 32, 32, 16), (2, 2, 3, 4), (5, 37, 24, 20)))
    def test_bmm_chain(self, shape):
        # Token mixing: O2[b] = (A[b] @ T[b] + R[b]) @ T[b]^T.
        def f(A, T, R):
            return torch.bmm(torch.bmm(A, T) + R, T.transpose(1, 2))

        b, n, l, d = shape
        args = (
            torch.randn(b, n, l, dtype=torch.float16),
            torch.randn(b, l, d, dtype=torch.float16),
            torch.randn(b, n, d, dtype=torch.float16),
        )
        code = self.check(f, args, fused=True)
        # Only the second product is stored; the first is read from registers.
        FileCheck().check("tl.dot").check("local_r0_index").check("tl.dot").run(code)
        FileCheck().check_count("tl.store", 1, exactly=True).run(code)

    def test_mlp(self):
        def f(x, w1, b1, w2, b2):
            return torch.relu(x @ w1 + b1) @ w2 + b2

        args = (
            torch.randn(128, 64, dtype=torch.float16),
            torch.randn(64, 64, dtype=torch.float16),
            torch.randn(64, dtype=torch.float16),
            torch.randn(64, 32, dtype=torch.float16),
            torch.randn(32, dtype=torch.float16),
        )
        self.check(f, args, fused=True)

    def test_padded_lanes(self):
        # 1 / (x @ w1) is inf past N = 20, where the second tl.dot reduces.
        def f(x, w1, w2):
            return (1.0 / (x @ w1)) @ w2

        args = (
            torch.randn(100, 64, dtype=torch.float16),
            torch.randn(64, 20, dtype=torch.float16),
            torch.randn(20, 24, dtype=torch.float16),
        )
        code = self.check(f, args, fused=True)
        FileCheck().check("tl.dot").check("tl.where(local_r0_index_mask").check(
            "tl.dot"
        ).run(code)

    def test_stored_intermediate(self):
        def f(A, T, R):
            mixed = torch.bmm(A, T)
            return torch.bmm(mixed + R, T.transpose(1, 2)), mixed

        args = (
            torch.randn(8, 32, 32, dtype=torch.float16),
            torch.randn(8, 32, 16, dtype=torch.float16),
            torch.randn(8, 32, 16, dtype=torch.float16),
        )
        self.check(f, args, fused=True)

    def test_epilogue_reading_outer_tile_not_fused(self):
        # The epilogue reads y in the [M, L] layout of the second product, but
        # the dot stage holds y with N on r0_.
        def f(x, w1, w2):
            y = x @ w1
            return y @ w2 + y

        args = (
            torch.randn(64, 32, dtype=torch.float16),
            torch.randn(32, 32, dtype=torch.float16),
            torch.randn(32, 32, dtype=torch.float16),
        )
        self.check(f, args, fused=False)

    @parametrize("dims", ((64, 64, 64, 32), (64, 128, 96, 64, 16)))
    def test_mlp_chain(self, dims):
        # Each product after the first starts a dot stage; equal widths give a
        # stage the same x and r0_ extents.
        def f(x, *params):
            weights, biases = params[0::2], params[1::2]
            for i, (weight, bias) in enumerate(zip(weights, biases)):
                x = x @ weight + bias
                if i < len(weights) - 1:
                    x = torch.relu(x)
            return x

        args = [torch.randn(128, dims[0], dtype=torch.float16)]
        for k, n in itertools.pairwise(dims):
            args.append(torch.randn(k, n, dtype=torch.float16) / k**0.5)
            args.append(torch.randn(n, dtype=torch.float16))
        code = self.check(f, tuple(args), fused=True)
        FileCheck().check_count("tl.dot", len(dims) - 1, exactly=True).run(code)

    def test_bmm_mlp_chain(self):
        def f(x, w1, w2, w3):
            hidden = torch.nn.functional.gelu(torch.bmm(x, w1))
            return torch.bmm(torch.nn.functional.gelu(torch.bmm(hidden, w2)), w3)

        args = (
            torch.randn(16, 32, 64, dtype=torch.float16),
            *(torch.randn(16, 64, 64, dtype=torch.float16) / 8 for _ in range(3)),
        )
        self.check(f, args, fused=True)

    def test_residual_across_stages_not_fused(self):
        # The second stage's epilogue reads h1, which only the first stage
        # holds in registers, with N on r0_.
        def f(x, w1, w2, w3):
            h1 = x @ w1
            return (torch.relu(h1) @ w2 + h1) @ w3

        args = (
            torch.randn(64, 32, dtype=torch.float16),
            torch.randn(32, 32, dtype=torch.float16) / 6,
            torch.randn(32, 32, dtype=torch.float16) / 6,
            torch.randn(32, 16, dtype=torch.float16),
        )
        self.check(f, args, fused=False)

    @parametrize("case", ("wide_n", "wide_l", "long_k", "transposed"))
    def test_not_fused(self, case):
        def f(x, w1, w2):
            y = x @ w1
            return (y.T if case == "transposed" else y) @ w2

        m, k, n, l = {
            "wide_n": (64, 64, 256, 32),
            "wide_l": (64, 64, 32, 256),
            "long_k": (64, 1024, 32, 32),
            "transposed": (64, 32, 64, 16),
        }[case]
        args = (
            torch.randn(m, k, dtype=torch.float16),
            torch.randn(k, n, dtype=torch.float16) / k**0.5,
            torch.randn(m if case == "transposed" else n, l, dtype=torch.float16),
        )
        self.check(f, args, fused=False)


class TestNativeMatmulRowReductionScheduling(TestCase):
    @parametrize("coalesce", (False, True))
    def test_benchmark_fusion(self, device, coalesce):
        def f(x, weight):
            return (x @ weight).sum(-1)

        args = (
            torch.randn(16, 16, device=device, dtype=torch.float16),
            torch.randn(16, 8, device=device, dtype=torch.float16),
        )
        torch._dynamo.reset()
        with (
            fresh_inductor_cache(),
            inductor_config.patch(
                {
                    "triton.native_matmul": True,
                    "benchmark_fusion": True,
                    "triton.coalesce_tiling_analysis": coalesce,
                }
            ),
        ):
            actual, code = run_and_get_code(torch.compile(f), *args)
        self.assertEqual(actual, f(*args), atol=5e-2, rtol=5e-2)
        FileCheck().check_count("@triton.jit", 1, exactly=True).check("tl.dot").check(
            "tl.sum"
        ).run(code[0])

    @parametrize("per_subkernel_blocks", (False, True))
    @parametrize("autotune", (0, 1))
    def test_combo_kernel(self, device, per_subkernel_blocks, autotune):
        def f(x, weight, z):
            return (x @ weight).sum(-1), z.sin()

        args = (
            torch.randn(16, 16, device=device, dtype=torch.float16),
            torch.randn(16, 8, device=device, dtype=torch.float16),
            torch.randn(137, device=device),
        )
        torch._dynamo.reset()
        with (
            fresh_inductor_cache(),
            inductor_config.patch(
                {
                    "triton.native_matmul": True,
                    "combo_kernels": True,
                    "combo_kernels_autotune": autotune,
                    "combo_kernel_peak_memory_pct_threshold": None,
                    "combo_kernel_per_subkernel_blocks": per_subkernel_blocks,
                }
            ),
        ):
            actual, code = run_and_get_code(torch.compile(f), *args)
        self.assertEqual(actual, f(*args), atol=5e-2, rtol=5e-2)
        FileCheck().check("tl.dot").check("tl.sum").run(code[0])

    @parametrize("n", (8, 128))
    @parametrize("per_subkernel_blocks", (False, True))
    @parametrize("autotune", (0, 1))
    def test_combo_kernel_complete_rows(
        self, device, n, per_subkernel_blocks, autotune
    ):
        def f(x, weight, z):
            return (x @ weight).sum(-1), (z @ weight).sum(-1)

        args = (
            torch.randn(16, 16, device=device, dtype=torch.float16),
            torch.randn(16, n, device=device, dtype=torch.float16),
            torch.randn(32, 16, device=device, dtype=torch.float16),
        )
        metrics.reset()
        torch._dynamo.reset()
        with (
            fresh_inductor_cache(),
            inductor_config.patch(
                {
                    "triton.native_matmul": True,
                    "combo_kernels": True,
                    "combo_kernels_autotune": autotune,
                    "combo_kernel_peak_memory_pct_threshold": None,
                    "combo_kernel_per_subkernel_blocks": per_subkernel_blocks,
                    "triton.coalesce_tiling_analysis": True,
                }
            ),
        ):
            actual, code = run_and_get_code(torch.compile(f), *args)
        self.assertEqual(actual, f(*args), atol=5e-2, rtol=5e-2)
        self.assertEqual(metrics.generated_kernel_count, 2)
        source = "\n".join(code)
        FileCheck().check_count("@triton.jit", 2, exactly=True).run(source)
        FileCheck().check_count("tl.dot", 2, exactly=True).run(source)
        FileCheck().check_count("tl.sum", 2, exactly=True).run(source)
        FileCheck().check_not("combo_grid_meta").run(source)


instantiate_device_type_tests(
    TestNativeMatmulRowReductionScheduling, globals(), only_for=GPU_TYPE
)


class TestNativeMatmulRowReductionHeuristic(TestCase):
    @parametrize("k", (511, 512, 513, 1024))
    @parametrize("m,n", ((256, 128), (4096, 64)))
    def test_matvec_k_limit(self, device, k, m, n):
        def f(x, weight, z):
            return (x @ weight) @ z

        x = torch.ones(m, k, device=device, dtype=torch.float16)
        weight = torch.full((k, n), 1 / k, device=device, dtype=torch.float16)
        z = torch.ones(n, device=device, dtype=torch.float16)
        metrics.reset()
        torch._dynamo.reset()
        with inductor_config.patch({"triton.native_matmul": True}):
            actual, _ = run_and_get_code(torch.compile(f), x, weight, z)
        self.assertEqual(actual, f(x, weight, z))
        self.assertEqual(metrics.generated_kernel_count, 1 if k <= 512 else 2)


instantiate_device_type_tests(
    TestNativeMatmulRowReductionHeuristic, globals(), only_for=GPU_TYPE
)


if HAS_GPU:
    torch.set_default_device(GPU_TYPE)

if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
