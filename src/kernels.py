"""
Kernels implemented with Triton
"""
import torch
import triton 
import triton.language as tl 

@triton.jit
def dyt_kernel_triton(
    x_ptr, 
    out_ptr, 
    alpha, 
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

def dyt_triton(x: torch.Tensor, alpha: float):
    n_elements = x.numel()
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]))

    dyt_kernel_triton[grid](
        x, out, alpha, n_elements, 
        BLOCK_SIZE=1024
    )
    return out