#pragma once
#include <torch/extension.h>

void dyt_launch_forward(torch::Tensor x, torch::Tensor out, torch::Tensor alpha, 
                torch::Tensor weight, torch::Tensor bias);
                
void dyt_launch_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor alpha,
                        torch::Tensor weight, torch::Tensor grad_x, torch::Tensor grad_alpha,
                        torch::Tensor grad_weight, torch::Tensor grad_bias);