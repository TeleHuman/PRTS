import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn.functional as F
from qwen_RoPE import fused_qwen_rope
from einops import rearrange

torch.manual_seed(42)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def torch_rope_ref(
    q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
):
    orig_q_dtype = q.dtype
    q= q.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    return q_embed


def test_accuracy():
    print("="*60)
    print("🔍 Running RoPE Accuracy Test (BF16 CUDA vs FP32 PyTorch)")
    print("="*60)

    BATCH_SIZE = 1
    SEQ_LEN = 53284
    NUM_HEADS = 16
    HEAD_DIM = 64
    EMBED_DIM = NUM_HEADS * HEAD_DIM
    DEVICE = "cuda"

    # --- 构造输入数据 ---
    x_fp32 = torch.randn(BATCH_SIZE,SEQ_LEN,NUM_HEADS,HEAD_DIM, device=DEVICE, dtype=torch.float32, requires_grad=True)
    
    # 构造 Freqs (Complex64)
    # 模拟 [S, 1, HeadDim//2]
    
    cos = torch.randn(BATCH_SIZE,SEQ_LEN,HEAD_DIM//2, device = "cuda",dtype=torch.float32)
    sin = torch.randn(BATCH_SIZE,SEQ_LEN,HEAD_DIM//2, device = "cuda",dtype=torch.float32)
    cos = torch.cat((cos,cos),dim=-1)
    sin = torch.cat((sin,sin),dim=-1)
    
    # 2. 构造 CUDA 输入 (BF16)
    x_bf16 = x_fp32.detach().to(torch.bfloat16).clone()
    x_bf16.requires_grad = True
    x_cuda = x_fp32.detach().to(torch.bfloat16).clone()
    x_cuda.requires_grad = True
    
    # 注意：freqs 在 Python 接口里是 complex64，底层 CUDA 也是读 complex64 (float2)
    # 不需要转 bf16
    
    # ==========================================
    # 1. 前向传播 (Forward) 测试
    # ==========================================
    
    
    # Run PyTorch Reference (FP32)
    ref_out = torch_rope_ref(x_fp32, cos,sin)
    
    init_out = torch_rope_ref(x_bf16, cos,sin) 
    # Run CUDA Custom Op (BF16)
    cuda_out = fused_qwen_rope(x_cuda, cos,sin)

    # 对比: 将 ref_out 转为 bf16，或者将 cuda_out 转为 fp32 对比
    diff_py = (ref_out.to(torch.bfloat16) - init_out).abs()
    max_diff_py = diff_py.max().item()
    mean_diff_py = diff_py.mean().item()

    print(f"\n[Forward Pass]")
    print(f"Ref Output (FP32) vs pytorch Output (BF16)")
    print(f"Max Diff: {max_diff_py:.6f}")
    print(f"Mean Diff: {mean_diff_py:.6f}")
    
    # 阈值判定：BF16 的有效位数较少，RoPE 包含乘加运算，累积误差在 1e-2 左右是正常的
    if max_diff_py < 0.2: 
        print("✅ Forward Pass CHECKED (Difference within expected BF16 range)")
    else:
        print("❌ Forward Pass FAILED (Difference too large)")
    
    # 对比: 将 ref_out 转为 bf16，或者将 cuda_out 转为 fp32 对比
    diff_cuda = (ref_out.to(torch.bfloat16) - cuda_out).abs()
    max_diff_cuda = diff_cuda.max().item()
    mean_diff_cuda = diff_cuda.mean().item()

    print(f"\n[Forward Pass]")
    print(f"Ref Output (FP32) vs CUDA Output (BF16)")
    print(f"Max Diff: {max_diff_cuda:.6f}")
    print(f"Mean Diff: {mean_diff_cuda:.6f}")
    
    # 阈值判定：BF16 的有效位数较少，RoPE 包含乘加运算，累积误差在 1e-2 左右是正常的
    if max_diff_cuda < 0.2: 
        print("✅ Forward Pass CHECKED (Difference within expected BF16 range)")
    else:
        print("❌ Forward Pass FAILED (Difference too large)")

    # ==========================================
    # 2. 反向传播 (Backward) 测试
    # ==========================================
    
    # 构造一个随机的梯度 (BF16)
    grad_output = torch.randn_like(x_cuda)
    
    # PyTorch Ref Backward
    # 需要先将 grad 转为 fp32 喂给 ref
    ref_loss = (ref_out * grad_output.float()).sum()
    ref_loss.backward()
    grad_ref = x_fp32.grad
    
    # CUDA Custom Op Backward
    cuda_loss = (cuda_out * grad_output).sum()
    cuda_loss.backward()
    grad_cuda = x_cuda.grad

    py_loss = (init_out * grad_output).sum()
    py_loss.backward()
    grad_py = x_bf16.grad
    
    # 对比梯度
    grad_diff_py = (grad_ref - grad_py.float()).abs()
    max_grad_diff_py = grad_diff_py.max().item()
    mean_grad_diff_py = grad_diff_py.mean().item()

    print(f"\n[Backward Pass]")
    print(f"Ref Grad (FP32) vs PyTorch Grad (BF16)")
    print(f"Max Grad Diff: {max_grad_diff_py:.6f}")
    print(f"Mean Grad Diff: {mean_grad_diff_py:.6f}")

    if max_grad_diff_py < 0.1:
        print("✅ Backward Pass CHECKED")
    else:
        print("❌ Backward Pass FAILED")
        
    # 对比梯度
    grad_diff = (grad_ref - grad_cuda.float()).abs()
    max_grad_diff = grad_diff.max().item()
    mean_grad_diff = grad_diff.mean().item()
    
    print(f"\n[Backward Pass]")
    print(f"Ref Grad (FP32) vs CUDA Grad (BF16)")
    print(f"Max Grad Diff: {max_grad_diff:.6f}")
    print(f"Mean Grad Diff: {mean_grad_diff:.6f}")

    if max_grad_diff < 0.1:
        print("✅ Backward Pass CHECKED")
    else:
        print("❌ Backward Pass FAILED")

if __name__ == "__main__":
    test_accuracy()