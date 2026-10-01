"""
Jagged views over packed storage.

    x   = torch.ops.jagged.view(values, offsets)     # values [NNZ, *D], offsets [B + 1]  ->  x [B, J, *D]
    out = torch.ops.jagged.values(y, offsets)        # y [B, J, *D]                       ->  out [NNZ, *D]

J is a fresh size symbol: row b has J(b) = offsets[b + 1] - offsets[b] valid entries. Tracing sees x as a plain
dense tensor, so every op in between is an ordinary aten op on a [B, J, ...] shape. The symbol is recorded in
ShapeEnv.jagged_symbols (J -> (depend_dim, NNZ)) so graphs print it as J(0) and lowering can find the storage.

Eager (for reference only) pads with zeros and compacts back.
"""

import torch


@torch.library.custom_op("jagged::view", mutates_args=())
def view(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    lengths = offsets.diff()
    B, J = lengths.numel(), int(lengths.max()) if lengths.numel() else 0
    row = torch.repeat_interleave(torch.arange(B, device=values.device), lengths)
    col = torch.arange(values.shape[0], device=values.device) - offsets[:-1][row]
    out = values.new_zeros(B, J, *values.shape[1:])
    out[row, col] = values
    return out


@view.register_fake
def _(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    J = torch.library.get_ctx().new_dynamic_size(min=1)
    J.node.shape_env.jagged_symbols[J.node.expr] = (0, values.shape[0])
    return values.new_empty(offsets.shape[0] - 1, J, *values.shape[1:])


@torch.library.custom_op("jagged::values", mutates_args=())
def values(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    lengths = offsets.diff()
    mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
    return x[mask]


@values.register_fake
def _(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    J = x.shape[1]
    _, nnz = J.node.shape_env.jagged_symbols[J.node.expr]
    return x.new_empty(nnz, *x.shape[2:])
