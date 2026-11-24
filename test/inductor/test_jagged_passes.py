# Owner(s): ["module: inductor"]

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
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


instantiate_device_type_tests(TestLowerFbgemmJaggedOps, globals(), allow_xpu=True)


if __name__ == "__main__":
    run_tests()
