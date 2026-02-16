/*
DyT implemented with CUDA
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void dytKernel(const float *__restrict__ x, float *__restrict__ out,
                           const float *__restrict__ alpha,
                           const float *__restrict__ weight,
                           const float *__restrict__ bias,
                           const int num_features, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4; // vectorized, float4

    if (vec_idx < n) {
        float4 x_vec;
        float4 out_vec;
        float a = alpha[0];

        if (vec_idx + 4 <= n) {
            x_vec = reinterpret_cast<const float4 *>(x)[idx];
        } else {
            x_vec.x = x[vec_idx];
            if (vec_idx + 1 < n) x_vec.y = x[vec_idx + 1];
            if (vec_idx + 2 < n) x_vec.z = x[vec_idx + 2];
        }


        // process 4 elements
        #pragma unroll
        for (int i = 0; i < 4; i++) { // pragma helps extend this loop
            float x_val;
            if (i == 0) x_val = x_vec.x;
            else if (i == 1) x_val = x_vec.y;
            else if (i == 2) x_val = x_vec.z;
            else x_val = x_vec.w;

            int curr_idx = vec_idx + i;
            if (curr_idx < n) {
                size_t feat_idx = curr_idx % num_features;
                float activated = tanhf(x_val * a);
                float result = __fmaf_rn(activated, weight[feat_idx], bias[feat_idx]);

                if (i == 0) out_vec.x = result;
                else if (i == 1) out_vec.y = result;
                else if (i == 2) out_vec.z = result;
                else out_vec.w = result;
            }
        }
    
        if (vec_idx + 4 <= n) {
            reinterpret_cast<float4 *>(out)[idx] = out_vec;
        } else {
            for (int i = 0; i < n - vec_idx; i++) {
                float res;
                if (i == 0) res = out_vec.x;
                else if (i == 1) res = out_vec.y;
                else if (i == 2) res = out_vec.z;
                else res = out_vec.w;
                out[vec_idx + i] = res;
            }
        }
    }
}

void dyt_launch(torch::Tensor x, torch::Tensor out,
                torch::Tensor alpha, torch::Tensor weight, torch::Tensor bias) {
    int n = x.numel();
    int num_features = x.size(-1); // feat_dim at -1
    int threads = 256;
    int vec_n = (n + 3) / 4;
    int blocks = (vec_n + threads - 1) / threads;

    // launch the kernel
    dytKernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(),
                                alpha.data_ptr<float>(),
                                weight.data_ptr<float>(),
                                bias.data_ptr<float>(), num_features, n);
}