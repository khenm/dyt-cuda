"""
Benchmarking 3 implementation: torch-based, triton and cuda.
This benchmark only works on CUDA devices. 
"""
import os
import random
import numpy as np
import argparse
import torch
import triton 
import triton.language as tl 
import triton.testing
from src.kernels import dyt_triton
from src.torch import DyT
import dyt_cuda

def seed_everything(seed: int = 42):
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_inputs(N, num_features, device='cuda'):
    x = torch.randn((N, num_features), device=device, dtype=torch.float32)
    model = DyT(num_features=num_features).to(device)
    alpha = model.alpha.detach()
    weight = model.weight.detach()
    bias = model.bias.detach()
    return x, model, alpha, weight, bias


@triton.testing.perf_report([
    triton.testing.Benchmark( # n-scaling
        x_names=['N'],
        x_vals=[128 * 2 ** i for i in range(0, 10)],
        line_arg='provider',
        line_vals=['torch', 'triton', 'cuda'],
        line_names=['PyTorch', 'Triton', 'CUDA'],
        plot_name='dyt-performance-n-scaling',
        styles=[('#4988C4', ':'), ('#1C4D8D', '--'), ('#0F2854', '-')],
        ylabel='Throughput (Gb/s)',
        xlabel='Input Size (N)',
        args={'num_features': 4096}
    ), 
    triton.testing.Benchmark( # feature-scaling
        x_names=['num_features'],
        x_vals=[512 * 2 ** i for i in range(0, 8)], 
        line_arg='provider', 
        line_vals=['torch', 'triton', 'cuda'], 
        line_names=['PyTorch', 'Triton', 'CUDA'],
        styles=[('#4988C4', ':'), ('#1C4D8D', '--'), ('#0F2854', '-')],
        ylabel='Throughput (GB/s)', 
        xlabel='Feature Size',
        plot_name='dyt-performance-feature-scaling',
        args={'N': 4096}
    )
])
def benchmark(N, num_features, provider):
    x, model, alpha, weight, bias = get_inputs(N, num_features)
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
       ms, min_ms, max_ms = triton.testing.do_bench(lambda: model(x), quantiles=quantiles)
    elif provider == 'triton':
       ms, min_ms, max_ms = triton.testing.do_bench(lambda: dyt_triton(x, alpha, weight, bias), quantiles=quantiles)
    elif provider == 'cuda':
       ms, min_ms, max_ms = triton.testing.do_bench(lambda: dyt_cuda.forward(x, alpha, weight, bias), quantiles=quantiles)

    def gbps(ms):
       total_bytes = x.numel() * x.element_size() * 2
       gbps = total_bytes * 1e-9 / (ms * 1e-3)
       return gbps
   
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DyT Benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./results")
    args = parser.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    benchmark.run(print_data=True, show_plots=False, save_path=args.save_dir) # turn off to save plots
    