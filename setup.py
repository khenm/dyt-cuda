from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            name='dyt_cuda', 
            sources=[
                'src/bindings.cpp',
                'src/kernels.cu'
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)