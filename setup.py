from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dyt',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name="dyt_cuda", 
            sources=[
                "csrc/bindings.cpp", 
                "csrc/kernels.cu"
            ], 
            extra_compile_args={
                'cxx': ['-03'],
                'nvcc': ['-03']
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
