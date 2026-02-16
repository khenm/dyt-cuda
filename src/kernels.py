"""
DyT implemented with Triton
"""
import torch
import triton 
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def dyt_kernel(
    x_ptr, 
    out_ptr, 
    alpha_ptr, 
    weight_ptr,
    bias_ptr,
    num_features,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    alpha = tl.load(alpha_ptr)
    feat_offsets = offsets % num_features
    weight = tl.load(weight_ptr + feat_offsets, mask=mask)
    bias = tl.load(bias_ptr + feat_offsets, mask=mask)
    
    activated = libdevice.tanh(x * alpha)
    result = activated * weight + bias
    tl.store(out_ptr + offsets, result, mask=mask)
    

def dyt_triton(x: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    n_elements = x.numel()
    num_features = x.size(-1)
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    dyt_kernel[grid](
        x, out, alpha,
        weight, bias,
        num_features,
        n_elements,
        BLOCK_SIZE=1024
    )
    return out