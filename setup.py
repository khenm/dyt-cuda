from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dyt',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name="dyt", 
            sources=[
                "csrc/bindings.cpp", 
                "csrc/kernels.cu"
            ], 
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
