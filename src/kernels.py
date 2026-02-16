"""
DyT implemented with Triton
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def dyt_forward_kernel(
    x_ptr,
    out_ptr,
    alpha_ptr,
    weight_ptr,
    bias_ptr,
    num_features,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
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
    
@triton.jit
def dyt_backward_kernel(
    grad_out_ptr,
    x_ptr,
    alpha_ptr, 
    weight_ptr,
    grad_x_ptr,
    grad_alpha_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    num_features,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    alpha = tl.load(alpha_ptr)
    feat_offsets = offsets % num_features
    weight = tl.load(weight_ptr + feat_offsets, mask=mask, other=0.0)
    
    z = x * alpha
    a = libdevice.tanh(z)
    dtanh = 1.0 - (a * a)
    
    grad_x = grad_out * weight * dtanh * alpha
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
    
    tl.atomic_add(grad_bias_ptr + feat_offsets, grad_out, mask=mask)
    
    grad_w_ = grad_out * a
    tl.atomic_add(grad_weight_ptr + feat_offsets, grad_w_, mask=mask)
    
    grad_a = grad_out * weight * dtanh * x
    grad_a_masked = tl.where(mask, grad_a, 0.0)
    block_alpha_sum = tl.sum(grad_a_masked)
    tl.atomic_add(grad_alpha_ptr, block_alpha_sum)


def dyt_triton_forward(
    x: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
):
    n_elements = x.numel()
    num_features = x.size(-1)
    out = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    dyt_forward_kernel[grid](
        x, out, alpha, weight, bias, num_features, n_elements, BLOCK_SIZE=1024
    )
    return out

def dyt_triton_backward(
    grad_out: torch.Tensor, x: torch.Tensor, alpha: torch.Tensor, weight: torch.Tensor
):
    n_elements = x.numel()
    num_features = x.size(-1)
    grad_x = torch.empty_like(x)
    grad_alpha = torch.zeros_like(alpha)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(weight)
    
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    dyt_backward_kernel[grid](
        grad_out, x, alpha, weight, grad_x, grad_alpha, grad_weight, grad_bias, num_features, n_elements, BLOCK_SIZE=1024
    )
    return grad_x, grad_alpha, grad_weight, grad_bias
                
    
