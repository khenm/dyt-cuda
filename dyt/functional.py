"""
Forward and Backward of DyT
"""
import torch
import torch.nn as nn
from .kernels import dyt_triton_forward, dyt_triton_backward
import dyt_cuda

class DyTFunctionCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, weight, bias):
        ctx.save_for_backward(x, alpha, weight, bias)
        return dyt_cuda.forward(x, alpha, weight, bias)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, weight, bias = ctx.saved_tensors
        return dyt_cuda.backward(grad_output, x, alpha, weight)

class DyTFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, weight, bias):
        ctx.save_for_backward(x, alpha, weight, bias)
        return dyt_triton_forward(x, alpha, weight, bias)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, weight, bias = ctx.saved_tensors
        return dyt_triton_backward(grad_output, x, alpha, weight)