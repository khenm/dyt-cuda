/*
CUDA bindings
*/
#include <torch/extension.h>

void dyt_launch_forward(torch::Tensor x, torch::Tensor out, torch::Tensor alpha, 
                torch::Tensor weight, torch::Tensor bias);
void dyt_launch_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor alpha,
                        torch::Tensor weight, torch::Tensor grad_x, torch::Tensor grad_alpha,
                        torch::Tensor grad_weight, torch::Tensor grad_bias);

torch::Tensor dyt_forward(torch::Tensor x, torch::Tensor alpha,
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
    dyt_launch_forward(x_c, out, alpha_c, weight_c, bias_c);
    return out;
}

std::vector<torch::Tensor> dyt_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor alpha,
                        torch::Tensor weight) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(alpha.is_cuda(), "alpha must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

    auto grad_output_c = grad_output.contiguous();
    auto x_c = x.contiguous();
    auto alpha_c = alpha.contiguous();
    auto weight_c = weight.contiguous();

    auto grad_x = torch::zeros_like(x_c);
    auto grad_alpha = torch::zeros_like(alpha_c);
    auto grad_weight = torch::zeros_like(weight_c);
    auto grad_bias = torch::zeros_like(weight_c);

    dyt_launch_backward(grad_output_c, x_c, alpha_c, weight_c, grad_x, grad_alpha, grad_weight, grad_bias);
    return {grad_x, grad_alpha, grad_weight, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dyt_forward, "DYT Forward (CUDA)");
    m.def("backward", &dyt_backward, "DYT Backward (CUDA)");
}