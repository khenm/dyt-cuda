/*
CUDA bindings
*/
#include <torch/extension.h>

void dyt_launch(torch::Tensor x, torch::Tensor out, float alpha);

void dyt_forward(torch::Tensor x, torch::Tensor out, float alpha) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    dyt_launch(x, out, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward". &dyt_forward, "DYT Forward (CUDA)");
}