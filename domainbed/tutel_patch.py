"""Pure-PyTorch fallbacks for Tutel routing kernels."""

import torch



def _fast_cumsum_sub_one(data: torch.Tensor) -> torch.Tensor:
    """Return the zero-based cumulative sum along dimension 0."""
    return data.long().cumsum(dim=0) - 1


# Sparse scatter and gather replacements.

def _get_gate_scalar(gates: torch.Tensor, valid_idx: torch.Tensor) -> torch.Tensor:
    """Extract valid gate values, including the two-column format."""
    if gates.dim() == 2:
        return gates[valid_idx, 0]
    return gates[valid_idx]


def _make_pytorch_fwd(dtype, is_cuda=True):
    def func_fwd(gates, indices, locations, src, dst, extra):
        N, _hidden, capacity = extra
        src2d = src.reshape(-1, src.size(-1))
        dst2d = dst.reshape(-1, dst.size(-1))
        mask = (locations < capacity) & (indices >= 0)
        valid = mask.nonzero(as_tuple=True)[0]
        if valid.numel() == 0:
            return
        slots = (indices[valid].long() * capacity + locations[valid].long())
        g = _get_gate_scalar(gates, valid).to(src2d.dtype).unsqueeze(1)
        dst2d.index_add_(0, slots, (g * src2d[valid]).to(dst2d.dtype))
    return func_fwd


def _make_pytorch_bwd_data(dtype, is_cuda=True):
    def func_bwd_data(gates, indices, locations, grad_src, dispatched, extra):
        N, _hidden, capacity = extra
        dispatched2d = dispatched.reshape(-1, dispatched.size(-1))
        grad2d = grad_src.reshape(-1, grad_src.size(-1))
        mask = (locations < capacity) & (indices >= 0)
        valid = mask.nonzero(as_tuple=True)[0]
        grad2d.zero_()
        if valid.numel() == 0:
            return
        slots = (indices[valid].long() * capacity + locations[valid].long())
        g = _get_gate_scalar(gates, valid).to(dispatched2d.dtype).unsqueeze(1)
        grad2d[valid] = (g * dispatched2d[slots]).to(grad2d.dtype)
    return func_bwd_data


def _make_pytorch_bwd_gate(dtype, is_cuda=True):
    def func_bwd_gate(grad_gates, indices, locations, src, dispatched, extra):
        N, _hidden, capacity = extra
        dispatched2d = dispatched.reshape(-1, dispatched.size(-1))
        src2d = src.reshape(-1, src.size(-1))
        mask = (locations < capacity) & (indices >= 0)
        valid = mask.nonzero(as_tuple=True)[0]
        grad_gates.zero_()
        if valid.numel() == 0:
            return
        slots = (indices[valid].long() * capacity + locations[valid].long())
        grad_gates[valid] = (src2d[valid].to(dispatched2d.dtype) * dispatched2d[slots]).sum(dim=1).to(grad_gates.dtype)
    return func_bwd_gate



def _apply_tutel_patch():
    """Replace Tutel CUDA routing functions with PyTorch equivalents."""
    try:
        from tutel.jit_kernels import sparse as _sparse
        _sparse.create_forward       = _make_pytorch_fwd
        _sparse.create_backward_data = _make_pytorch_bwd_data
        _sparse.create_backward_gate = _make_pytorch_bwd_gate
    except ImportError:
        pass

    try:
        from tutel.jit_kernels import gating as _gating
        _gating.fast_cumsum_sub_one = _fast_cumsum_sub_one
    except ImportError:
        pass

    try:
        from tutel.impls import fast_dispatch as _fd
        _fd.fast_cumsum_sub_one = _fast_cumsum_sub_one
    except ImportError:
        pass

    try:
        from tutel.impls.fast_dispatch import TutelMoeFastDispatcher
        TutelMoeFastDispatcher.kernel_pool.clear()
    except Exception:
        pass


_apply_tutel_patch()
