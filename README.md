# Kernels implementation of Dynamic Tanh (DyT)

Efficient Triton and CUDA implementations of Dynamic Tanh (DyT).

### Description

Dynamic Tanh (DyT) is a normalization-free activation function proposed as a replacement for LayerNorm/RMSNorm in transformer architectures. It applies a learnable scaling factor to the input before a Tanh nonlinearity, followed by an affine transformation.

$$
DyT(x) = \tanh(\alpha \cdot x) \cdot w + b
$$

referenced the following paper: [Transformers without Normalization](https://arxiv.org/abs/2503.10622). CVPR 2025. [Jiachen Zhu](https://jiachenzhu.github.io), [Xinlei Chen](https://xinleic.xyz/), [Kaiming He](https://people.csail.mit.edu/kaiming/), [Yann LeCun](http://yann.lecun.com) and [Zhuang Liu](https://liuzhuang13.github.io). FAIR, NYU, MIT, Princeton [[`arXiv`](https://arxiv.org/abs/2503.10622)][[`project page`](https://jiachenzhu.github.io/DyT/)]

### Benchmark Results
We benchmark the forward and backward throughput of Cuda DyT compared with Triton and Pytorch efficiency. 
| | |
| :---: | :---: |
| **Forward N-Scaling**<br>![Forward N-Scaling](results/dyt-performance-n-scaling.png) | **Forward Feature-Scaling**<br>![Forward Feature-Scaling](results/dyt-performance-feature-scaling.png) |
| **Backward N-Scaling**<br>![Backward N-Scaling](results/dyt-backward-n-scaling.png) | **Backward Feature-Scaling**<br>![Backward Feature-Scaling](results/dyt-backward-feature-scaling.png) |

### Pretraining Benchmark

We benchmark the model wav2vec 2.0 pre-training efficiency using DyT. The benchmark compares the training step time across different implementations (PyTorch, Triton, CUDA) against a baseline LayerNorm implementation. This study is conducted on an *NVIDIA RTX 5090 32GB* with a fixed training steps of 200 and seed 42.

**Effect of Batch Size w. Fixed Seq. Len. = 5s**

| | | |
| :---: | :---: | :---: |
| **Batch Size 8**<br>![Batch 8](results/dyt_benchmark_results_batch8_seq5.png) | **Batch Size 16**<br>![Batch 16](results/dyt_benchmark_results_batch16_seq5.png) | **Batch Size 32**<br>![Batch 32](results/dyt_benchmark_results_batch32_seq5.png) |

**Effect of Seq. Len. w Fixed Batch Size of 4**

| | | 
| :---: | :---: | 
| **Seq Len 10s**<br>![Seq 10s](results/dyt_benchmark_results_batch4_seq10.png) | **Seq Len 20s**<br>![Seq 20s](results/dyt_benchmark_results_batch4_seq20.png) |
| **Seq Len 40s**<br>![Seq 40s](results/dyt_benchmark_results_batch4_seq40.png) | **Seq Len 60s**<br>![Seq 60s](results/dyt_benchmark_results_batch4_seq60.png) |


### LICENSE

This project is licensed under the MIT License. Please refer to the [LICENSE](LICENSE) file for more details.
