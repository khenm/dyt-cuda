from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="dyt_cuda",
    ext_modules=[
        CUDAExtension('dyt_cuda', [
            'src/bindings.cpp',
            'src/kernels.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)