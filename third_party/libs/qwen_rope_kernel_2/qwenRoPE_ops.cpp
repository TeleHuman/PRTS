#include <torch/extension.h>

torch::Tensor apply_rope_vision_forward_cuda(torch::Tensor x, torch::Tensor cos, torch::Tensor sin);
torch::Tensor apply_rope_vision_backward_cuda(torch::Tensor grad, torch::Tensor cos, torch::Tensor sin);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_rope_forward", &apply_rope_vision_forward_cuda, "Apply qwen RoPE Forward (CUDA)");
    m.def("torch_launch_rope_backward", &apply_rope_vision_backward_cuda, "Apply qwen RoPE Backward (CUDA)");
}