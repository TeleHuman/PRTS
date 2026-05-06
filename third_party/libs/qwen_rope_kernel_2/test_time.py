import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn.functional as F
from qwen_RoPE import fused_qwen_rope
from einops import rearrange
import time

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

def benchmark_func(name, func, x, cos, sin, num_iters=1000, backward=False):

    for _ in range(50):
        y = func(x, cos, sin)
        if backward:
            y.sum().backward()
            x.grad = None
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        y = func(x, cos, sin)
        if backward:
            loss = y.sum()
            loss.backward()
            x.grad = None 
    torch.cuda.synchronize()
    
    end = time.time()
    avg_time = (end - start) / num_iters * 1000 # 转换为 ms
    return avg_time

def test_performance():
    print("="*60)
    print("🚀 Running RoPE Full Performance Benchmark (Forward & Backward)")
    print("="*60)

    # --- 配置参数 ---
    BATCH_SIZE = 8       
    SEQ_LEN = 2048    
    NUM_HEADS = 32
    HEAD_DIM = 128
    EMBED_DIM = NUM_HEADS * HEAD_DIM
    DEVICE = "cuda"
    ITERS = 1000        
    print(f"Config: Shape=[{BATCH_SIZE}, {SEQ_LEN}, {NUM_HEADS}, {HEAD_DIM}], dtype=bf16")


    x = torch.randn(SEQ_LEN, NUM_HEADS,HEAD_DIM, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    
    cos = torch.randn(SEQ_LEN,HEAD_DIM//2, device = "cuda",dtype=torch.float32)
    sin = torch.randn(SEQ_LEN,HEAD_DIM//2, device = "cuda",dtype=torch.float32)
    cos = torch.cat((cos,cos),dim=-1)
    sin = torch.cat((sin,sin),dim=-1)
    # ==========================================
    # 1. 测试 Forward Time
    # ==========================================
    print("\n[Phase 1] Benchmarking Forward Pass...")
    
    # PyTorch Forward
    py_fwd_time = benchmark_func("PyTorch Fwd", torch_rope_ref, x, cos, sin, ITERS, backward=False)
    
    # CUDA Forward
    cuda_fwd_time = benchmark_func("CUDA Fwd", fused_qwen_rope, x, cos, sin, ITERS, backward=False)

    print(f"PyTorch Forward: {py_fwd_time:.3f} ms")
    print(f"CUDA Forward:    {cuda_fwd_time:.3f} ms")
    print(f"Forward Speedup: {py_fwd_time / cuda_fwd_time:.2f}x")

    # ==========================================
    # 2. 测试 Forward + Backward Time
    # ==========================================
    print("\n[Phase 2] Benchmarking Forward + Backward (Training Step)...")
    
    # PyTorch Full
    py_full_time = benchmark_func("PyTorch Full", torch_rope_ref, x, cos, sin, ITERS, backward=True)
    
    # CUDA Full
    cuda_full_time = benchmark_func("CUDA Full", fused_qwen_rope, x, cos, sin, ITERS, backward=True)

    print(f"PyTorch Total (Fwd+Bwd): {py_full_time:.3f} ms")
    print(f"CUDA Total (Fwd+Bwd):    {cuda_full_time:.3f} ms")
    print(f"Total Speedup:           {py_full_time / cuda_full_time:.2f}x")

    # ==========================================
    # 3. 推算 Backward Only Time
    # ==========================================
    print("\n[Phase 3] Estimated Backward Only Time (Total - Forward)...")
    
    py_bwd_time = py_full_time - py_fwd_time
    cuda_bwd_time = cuda_full_time - cuda_fwd_time
    
    py_bwd_time = max(0.001, py_bwd_time)
    cuda_bwd_time = max(0.001, cuda_bwd_time)

    print(f"PyTorch Backward: {py_bwd_time:.3f} ms")
    print(f"CUDA Backward:    {cuda_bwd_time:.3f} ms")
    print(f"Backward Speedup: {py_bwd_time / cuda_bwd_time:.2f}x")
    
    print("="*60)

if __name__ == "__main__":
    test_performance()