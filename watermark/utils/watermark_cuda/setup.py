from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="watermark_cuda",
    ext_modules=[
        CUDAExtension(
            "watermark_cuda",
            [
                "watermark_cuda.cpp",
                "watermark_cuda_kernel.cu",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
