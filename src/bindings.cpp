/*
CUDA bindings
*/
#include <torch/extension.h>

void dyt_launch(torch::Tensor x, torch::Tensor out, torch::Tensor alpha, 
                torch::Tensor weight, torch::Tensor bias);

void dyt_forward(torch::Tensor x, torch::Tensor alpha,
                torch::Tensor weight, torch::Tensor bias)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(alpha.is_cuda(), "alpha must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    auto x_c = x.contiguous();
    auto alpha_c = alpha.contiguous();
    auto weight_c = weight.contiguous();
    auto bias_c = bias.contiguous();

    auto out = torch::empty_like(x_c);
    dyt_launch(x_c, out, alpha_c, weight_c, bias_c);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dyt_forward, "DYT Forward (CUDA)");
}