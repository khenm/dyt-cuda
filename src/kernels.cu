/*
DyT implemented with CUDA
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void dytForwardKernel(const float *__restrict__ x, float *__restrict__ out,
                           const float *__restrict__ alpha,
                           const float *__restrict__ weight,
                           const float *__restrict__ bias,
                           const int num_features, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a = alpha[0];
    int n_vec = n / 4;

    if (idx < n_vec) {
        float4 x_vec = reinterpret_cast<const float*>(x)[idx];
        float4 out_vec;

        // process 4 elements
        #pragma unroll
        for (int i = 0; i < 4; i++) { // pragma helps extend this loop
            float x_val;
            if (i == 0) x_val = x_vec.x;
            else if (i == 1) x_val = x_vec.y;
            else if (i == 2) x_val = x_vec.z;
            else x_val = x_vec.w;

            int flat_idx = idx * 4 + i;
            int feat_idx = flat_idx % num_features;
            float activated = tanhf(x_val * a);
            float res = __fmaf_rn(activated, weight[feat_idx], bias[feat_idx]);

            if (i == 0) out_vec.x = res;
            else if (i == 1) out_vec.y = res;
            else if (i == 2) out_vec.z = res;
            else out_vec.w = res;
        }
    
        if (idx == n_vec) {
            int start_idx = n_vec * 4;
            for (int i = 0; i < (n - start_idx); i++) {
                int curr_idx = start_idx + i;
                int feat_idx = curr_idx % num_features;

                float x_val = x[curr_idx];
                float activated = tanhf(x_val * a);
                float res = __fmaf_rn(activated, weight[feat_idx], bias[feat_idx]);
                out[curr_idx] = res;
            }
        }
    }
}

__global__ void dytBackwardKernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ x, 
    const float* __restrict__ alpha,
    const float* __restrict__ weight,
    float* __restrict__ grad_x, 
    float* __restrict__ grad_alpha,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    const int num_features,
    const int n
) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    extern __shared__ float shared[];
    float grad_alpha_shared = 0.0f;

    if (idx < n) {
        float g_out = grad_output[idx];
        float x_val = x[idx];
        float a = alpha[0];
        
        int feat_idx = idx % num_features;
        float w = weight[feat_idx];

        float z = x_val * a;
        float tanhz = tanhf(z);
        float dtanh = 1.0f - (tanhz * tanhz); // derivative of tanh

        // grad bias accum
        atomicAdd(&grad_bias[feat_idx], g_out);

        // grad weight accum, dl/dw = grad_out * tanh(alpha * x)
        float g_w = g_out * tanhz;
        atomicAdd(&grad_weight[feat_idx], g_w);

        // grad x accum, dl/dx = grad_out * w * (1 - tanh^2) * a
        float g_x = g_out * w * dtanh * a;
        grad_x[idx] = g_x;
        
        // grad alpha accum, dl/da = grad_out * w * (1 - tanh^2) * x
        grad_alpha_shared = g_out * w * dtanh * x_val;
    }
    shared[tid] = grad_alpha_shared;
    __syncthreads();

    // collapsed twofold tree
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&grad_alpha[0], shared[0]);
    }
}

void dyt_launch_forward(torch::Tensor x, torch::Tensor out,
                torch::Tensor alpha, torch::Tensor weight, torch::Tensor bias) {
    int n = x.numel();
    int num_features = x.size(-1); // feat_dim at -1
    int n_vec = n / 4;
    int total_threads_needed = n_vec + 1;

    int threads = 512;
    int blocks = (total_threads_needed + threads - 1) / threads;

    // launch the kernel
    dytForwardKernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(),
                                alpha.data_ptr<float>(),
                                weight.data_ptr<float>(),
                                bias.data_ptr<float>(), num_features, n);
}

void dyt_launch_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor alpha,
                        torch::Tensor weight, torch::Tensor grad_x, torch::Tensor grad_alpha,
                        torch::Tensor grad_weight, torch::Tensor grad_bias) {
    int n = x.numel();
    int num_features = x.size(-1);
    int threads = 512;
    int blocks = (n + threads - 1) / threads;

    size_t shared_mem_size = threads * sizeof(float);

    dytBackwardKernel<<<blocks, threads, shared_mem_size>>>(grad_output.data_ptr<float>(), x.data_ptr<float>(),
                                alpha.data_ptr<float>(), weight.data_ptr<float>(),
                                grad_x.data_ptr<float>(), grad_alpha.data_ptr<float>(),
                                grad_weight.data_ptr<float>(), grad_bias.data_ptr<float>(),
                                num_features, n);
}