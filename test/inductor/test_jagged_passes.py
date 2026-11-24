# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification


try:
    import fbgemm_gpu  # noqa: F401
except Exception:
    pass

if not hasattr(torch.ops.fbgemm, "jagged_to_padded_dense_forward"):
    # Reference implementations with fbgemm_gpu's schemas, for running without it.
    _lib = torch.library.Library("fbgemm", "FRAGMENT")  # noqa: SCOPED_LIBRARY

    def _define(schema, fn):
        name = schema.split("(")[0]
        _lib.define(schema)
        _lib.impl(name, fn, "CompositeExplicitAutograd")
        torch.library.register_fake(f"fbgemm::{name}", fn, lib=_lib)

    def _jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
        flat = values.reshape(values.size(0), -1)
        dense = torch.ops.aten._jagged_to_padded_dense_forward(
            flat, offsets, max_lengths, padding_value
        )
        return dense.reshape(*dense.shape[:2], *values.shape[1:])

    def _dense_to_jagged(dense, offsets, total_L=None):
        flat = dense.reshape(*dense.shape[:2], -1)
        values = torch.ops.aten._padded_dense_to_jagged_forward(flat, offsets, total_L)
        return values.reshape(-1, *dense.shape[2:])

    def _complete_cumsum(t_in):
        return torch.nn.functional.pad(t_in.cumsum(0, dtype=t_in.dtype), (1, 0))

    _define(
        "jagged_to_padded_dense_forward(Tensor values, Tensor[] offsets, SymInt[] max_lengths, float padding_value=0) -> Tensor",
        _jagged_to_padded_dense,
    )
    _define(
        "jagged_to_padded_dense_backward(Tensor grad_output, Tensor[] offsets, SymInt total_L) -> Tensor",
        _dense_to_jagged,
    )
    _define(
        "dense_to_jagged_forward(Tensor dense, Tensor[] x_offsets, SymInt? total_L=None) -> Tensor",
        _dense_to_jagged,
    )
    _define("asynchronous_complete_cumsum(Tensor t_in) -> Tensor", _complete_cumsum)

    def _pack_segments(t_in, lengths, max_length):
        offsets = _complete_cumsum(lengths)
        return _jagged_to_padded_dense(t_in, [offsets], [max_length])

    def _pack_segments_backward(data, lengths, total_length, max_length):
        return _dense_to_jagged(data, [_complete_cumsum(lengths)], total_length)

    _define(
        "pack_segments(Tensor t_in, Tensor lengths, SymInt max_length) -> Tensor",
        _pack_segments,
    )
    _define(
        "pack_segments_backward(Tensor data, Tensor lengths, SymInt total_length, SymInt max_length) -> Tensor",
        _pack_segments_backward,
    )

fbgemm = torch.ops.fbgemm
j2pd = fbgemm.jagged_to_padded_dense_forward


class TestLowerFbgemmJaggedOps(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def check(self, fn, *args):
        expected = fn(*args)
        with config.patch(lower_fbgemm_jagged_ops=True):
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        self.assertEqual(actual, expected)
        calls = [
            line
            for line in "\n".join(code).splitlines()
            if "torch.ops.fbgemm." in line and not line.lstrip().startswith("#")
        ]
        self.assertEqual(calls, [])

    def test_jagged_to_padded_dense(self, device):
        values = torch.randn(10, 8, device=device)
        offsets = torch.tensor([0, 2, 5, 10], device=device)

        def fn(values, offsets):
            return j2pd(values * 2, [offsets], [4], 1.5) + 1

        self.check(fn, values, offsets)

    def test_jagged_to_padded_dense_1d_values(self, device):
        ids = 2**40 + torch.arange(10, device=device)
        offsets = torch.tensor([0, 2, 5, 10], device=device)

        def fn(ids, offsets):
            return j2pd(ids, [offsets], [6], -1.0)

        self.check(fn, ids, offsets)

    def test_dense_to_jagged(self, device):
        dense = torch.randn(3, 4, 8, device=device)
        offsets = torch.tensor([0, 2, 5, 9], device=device)

        def fn(dense, offsets):
            a = fbgemm.dense_to_jagged_forward(dense * 2, [offsets], 9)
            b = fbgemm.jagged_to_padded_dense_backward(dense, [offsets], 9)
            return a + b

        self.check(fn, dense, offsets)

    def test_complete_cumsum(self, device):
        lengths = torch.tensor([2, 0, 3, 5], device=device, dtype=torch.int32)
        values = torch.randn(10, 8, device=device)

        def fn(lengths, values):
            offsets = fbgemm.asynchronous_complete_cumsum(lengths)
            return offsets, j2pd(values, [offsets], [5])

        self.check(fn, lengths, values)

    def test_pack_segments(self, device):
        values = torch.randn(10, 8, device=device)
        lengths = torch.tensor([2, 0, 3, 5], device=device)
        dense = torch.randn(4, 5, 8, device=device)

        def fn(values, lengths, dense):
            offsets = fbgemm.asynchronous_complete_cumsum(lengths)
            packed = fbgemm.pack_segments(values * 2, lengths, 5)
            grad = fbgemm.pack_segments_backward(dense, lengths, 10, 5)
            return offsets, packed + 1, grad

        self.check(fn, values, lengths, dense)


@unittest.skipIf(not SM90OrLater, "bf16 _grouped_mm needs sm90")
class TestJaggedBmmToGroupedMm(TestCase):
    hw_classification = HardwareClassification.CUDA

    def check(self, fn, *args, rewritten=True):
        expected = fn(*args)
        with config.patch(jagged_bmm_to_grouped_mm=True):
            actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
        code = "\n".join(code)
        self.assertEqual("_grouped_mm" in code, rewritten)
        self.assertEqual("extern_kernels.bmm" in code, not rewritten)

    @staticmethod
    def inputs(lengths, heads, k=32, n=16):
        lengths = torch.tensor(lengths, device="cuda")
        values = torch.randn(
            int(lengths.sum()), heads * k, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(len(lengths), k, n, device="cuda", dtype=torch.bfloat16)
        return values, lengths, weight

    @staticmethod
    def fn(values, lengths, weight, T=4):
        B, K, N = weight.shape
        H = values.size(1) // K
        offsets = torch.nn.functional.pad(lengths.cumsum(0), (1, 0))
        padded = torch.ops.aten._jagged_to_padded_dense_forward(
            values, [offsets], [T], 0.0
        )
        out = torch.bmm(padded.view(B, T * H, K), weight).view(B, T, H * N)
        # offsets rebuilt from the same lengths, as separate fbgemm ops leave them
        offsets = torch.nn.functional.pad(lengths.cumsum(0), (1, 0))
        return torch.ops.aten._padded_dense_to_jagged_forward(
            out, [offsets], values.size(0)
        )

    def test_rewrite(self):
        # the 6-row sequence is longer than T=4: its last 2 output rows are zeros
        self.check(self.fn, *self.inputs([2, 0, 6, 3], heads=1))

    def test_rewrite_heads(self):
        self.check(self.fn, *self.inputs([2, 0, 6, 3], heads=2))

    def test_too_many_groups(self):
        self.check(self.fn, *self.inputs([1] * 1024, heads=1), rewritten=False)


instantiate_device_type_tests(TestLowerFbgemmJaggedOps, globals(), allow_xpu=True)


if __name__ == "__main__":
    run_tests()
