from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='qwen_rope_cuda', 
    include_dirs=["include"],  
    version='0.1.0',
    description='Optimized QWEN RoPE CUDA implementation with BF16 support',
    ext_modules=[
        CUDAExtension(
            name='qwen_rope_cuda', 
            sources=[
                "qwenRoPE_ops.cpp", 
                "qwenRoPE_forward.cu",  
                "qwenRoPE_backward.cu",
            ],
            extra_compile_args={"cxx": ["-O3"], 
                                'nvcc': [
                    '-O3', 
                    '-arch=sm_90',       
                    '-DENABLE_BF16',     
                    '--use_fast_math',    
                    '-gencode=arch=compute_90,code=sm_90'
                ]
            } 
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)