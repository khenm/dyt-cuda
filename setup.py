from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dyt_cuda',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name="dyt_cuda", 
            sources=[
                "csrc/bindings.cpp", 
                "csrc/kernels.cu",
            ], 
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            },
            include_dirs=["csrc"]
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
