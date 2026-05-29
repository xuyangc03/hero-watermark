#include <torch/extension.h>
#include <vector>

void launch_uniform_seeded_kernel(torch::Tensor output, const torch::Tensor &seeds);
void launch_uniform_seeded_indexed_kernel(
    torch::Tensor output,
    const torch::Tensor &seeds,
    const torch::Tensor &indices);

torch::Tensor uniform_seeded(
    torch::Tensor seeds,
    int64_t vocab_size,
    torch::Dtype dtype = torch::kFloat32)
{
    TORCH_CHECK(seeds.device().is_cuda(), "Seeds must be on CUDA");
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
    auto output = torch::empty({seeds.size(0), vocab_size}, options);
    launch_uniform_seeded_kernel(output, seeds);
    return output;
}

torch::Tensor uniform_seeded_indexed(
    torch::Tensor seeds,
    torch::Tensor indices,
    torch::Dtype dtype = torch::kFloat32)
{
    TORCH_CHECK(seeds.device().is_cuda(), "Seeds must be on CUDA");
    TORCH_CHECK(indices.device().is_cuda(), "Indices must be on CUDA");
    TORCH_CHECK(seeds.device() == indices.device(), "Seeds and indices must be on the same device");
    TORCH_CHECK(seeds.size(0) == indices.size(0), "Seeds and indices must have the same batch size");
    auto options = torch::TensorOptions().dtype(dtype).device(seeds.device());
    auto output = torch::empty({seeds.size(0)}, options);
    launch_uniform_seeded_indexed_kernel(output, seeds, indices);
    return output;
}

torch::Tensor randperm_seeded(
    torch::Tensor seeds,
    int64_t vocab_size)
{
    torch::Tensor random_scores = uniform_seeded(seeds, vocab_size, torch::kFloat32);

    torch::Tensor perm = torch::argsort(random_scores, /*dim=*/1, /*descending=*/false);

    return perm;
}

torch::Tensor randperm_seeded_indexed(
    torch::Tensor seeds,
    torch::Tensor indices,
    int64_t vocab_size)
{
    torch::Tensor random_scores = uniform_seeded(seeds, vocab_size, torch::kFloat32);
    torch::Tensor full_perm = torch::argsort(random_scores, 1, false);
    torch::Tensor indices_expanded = indices.unsqueeze(1);
    return full_perm.gather(1, indices_expanded).squeeze(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("uniform_seeded", &uniform_seeded, "Generate uniform random numbers with row-wise seeds");
    m.def("uniform_seeded_indexed", &uniform_seeded_indexed, "Generate uniform random numbers at specific indices");
    m.def("randperm_seeded", &randperm_seeded, "Generate random permutations with row-wise seeds");
    m.def("randperm_seeded_indexed", &randperm_seeded_indexed, "Get specific element from random permutations");
}
