#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <curand_kernel.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void uniform_seeded_kernel(
    scalar_t *output_data,
    const int64_t *seeds_data,
    int64_t batch_size,
    int64_t vocab_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * vocab_size;
    if (idx < total_elements)
    {
        int64_t row_idx = idx / vocab_size;
        int64_t col_idx = idx % vocab_size;
        uint64_t seed = seeds_data[row_idx];
        curandStatePhilox4_32_10_t state;
        curand_init(seed, col_idx, 0, &state);
        output_data[idx] = static_cast<scalar_t>(curand_uniform(&state));
    }
}

template <typename scalar_t>
__global__ void uniform_seeded_indexed_kernel(
    scalar_t *output_data,
    const int64_t *seeds_data,
    const int64_t *indices_data,
    int64_t batch_size)
{
    int64_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < batch_size)
    {
        uint64_t seed = seeds_data[row_idx];
        int64_t col_idx = indices_data[row_idx];
        curandStatePhilox4_32_10_t state;
        curand_init(seed, col_idx, 0, &state);
        output_data[row_idx] = static_cast<scalar_t>(curand_uniform(&state));
    }
}

void launch_uniform_seeded_kernel(
    torch::Tensor output,
    const torch::Tensor &seeds)
{
    int64_t batch_size = output.size(0);
    int64_t vocab_size = output.size(1);
    int64_t total_elements = batch_size * vocab_size;
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(),
        "uniform_seeded_kernel",
        ([&]
         { uniform_seeded_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
               output.data_ptr<scalar_t>(),
               seeds.data_ptr<int64_t>(),
               batch_size,
               vocab_size); }));
}

void launch_uniform_seeded_indexed_kernel(
    torch::Tensor output,
    const torch::Tensor &seeds,
    const torch::Tensor &indices)
{
    int64_t batch_size = output.size(0);
    const int threads_per_block = 256;
    const int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(),
        "uniform_seeded_indexed_kernel",
        ([&]
         { uniform_seeded_indexed_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
               output.data_ptr<scalar_t>(),
               seeds.data_ptr<int64_t>(),
               indices.data_ptr<int64_t>(),
               batch_size); }));
}
