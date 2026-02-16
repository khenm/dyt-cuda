/*
Kernels implemented with CUDA
*/
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void dyt_kernel(const float *x, float *out, float alpha, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = tanhf(x[idx] * alpha);
  }
}

void dyt_launch(torch::Tensor x, torch::Tensor out, float alpha) {
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // launch the kernel
    dyt_kernel <<< blocks, threads >>> (
        x.data_ptr<float>(), 
        out.data_ptr<float>(),
        alpha, 
        n
    )
}