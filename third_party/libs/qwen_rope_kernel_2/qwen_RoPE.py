import torch
import qwen_rope_cuda

class RoPEVisionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        """
        x: [S, H, D] or [B, S, H, D] or [B, H, S, D] (BF16)
        cos: [S, D] or [B, S, D] or [B, 1, S, D] or [B, S, 1, D](FP32)
        sin: [S, D] or [B, S, D] or [B, 1, S, D] or [B, S, 1, D](FP32)
        """
        if not x.is_contiguous(): x = x.contiguous()
        
        if not cos.is_contiguous(): cos = cos.contiguous()
        if not sin.is_contiguous(): sin = sin.contiguous()

        ctx.save_for_backward(cos, sin)
        
        return qwen_rope_cuda.torch_launch_rope_forward(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous(): grad_output = grad_output.contiguous()
        cos, sin = ctx.saved_tensors
        
        grad_input = qwen_rope_cuda.torch_launch_rope_backward(grad_output, cos, sin)
        
        return grad_input, None, None

def fused_qwen_rope(x, cos, sin):
    # torch.distributed.breakpoint()
    return RoPEVisionFunction.apply(x, cos, sin)