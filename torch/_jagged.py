"""
Jagged views over packed storage.

    x   = torch.ops.jagged.view(values, offsets)     # values [NNZ, *D], offsets [B + 1]  ->  x [B, J, *D]
    out = torch.ops.jagged.values(y, offsets)        # y [B, J, *D]                       ->  out [NNZ, *D]

J is a size symbol, one per offsets tensor: row b has J(b) = offsets[b + 1] - offsets[b] valid entries. Tracing
sees x as a plain dense tensor, so every op in between is an ordinary aten op on a [B, J, ...] shape, and J itself may
be used as a size (torch.arange(J), dense[:, :J], F.pad(x, (0, N - J))): it means the length of the row it is used
in. The symbol is recorded in ShapeEnv.jagged_symbols (J -> (depend_dim, NNZ)) so graphs print it as J(0) and
lowering can find the storage.

Eager (for reference only) pads with zeros and compacts back.
"""

import torch
from torch.utils.weak import WeakIdKeyDictionary


# fake offsets -> J: views over the same offsets share the jagged dim
_jagged_dims = WeakIdKeyDictionary()


@torch.library.custom_op("jagged::view", mutates_args=())
def view(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    lengths = offsets.diff()
    B, J = lengths.numel(), int(lengths.max()) if lengths.numel() else 0
    row = torch.repeat_interleave(torch.arange(B, device=values.device), lengths)
    col = torch.arange(values.shape[0], device=values.device) - offsets[:-1][row]
    out = values.new_zeros(B, J, *values.shape[1:])
    out[row, col] = values
    return out


def _jagged_symbols(J: torch.SymInt) -> dict:
    shape_env = J.node.shape_env
    if shape_env is None:
        raise RuntimeError(f"jagged size {J} has no ShapeEnv")
    return shape_env.jagged_symbols


@view.register_fake
def _(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    J = _jagged_dims.get(offsets)
    if J is None:
        J = _jagged_dims[offsets] = torch.library.get_ctx().new_dynamic_size(min=1)
        _jagged_symbols(J)[J.node.expr] = (0, values.shape[0])
    return values.new_empty(offsets.shape[0] - 1, J, *values.shape[1:])


@torch.library.custom_op("jagged::values", mutates_args=())
def values(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    lengths = offsets.diff()
    mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
    return x[mask]


@values.register_fake
def _(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    J = x.shape[1]
    if not isinstance(J, torch.SymInt):
        raise RuntimeError(f"jagged.values: dim 1 of x is not jagged: {x.shape}")
    _, nnz = _jagged_symbols(J)[J.node.expr]
    return x.new_empty(nnz, *x.shape[2:])
