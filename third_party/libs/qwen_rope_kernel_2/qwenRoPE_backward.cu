#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "qwenRoPE_kernel.cuh"

template<int HEAD_DIM>
__global__ void apply_rope_vision_backward_shfl_kernel(
    const void* __restrict__ grad_output,
    const void* __restrict__ cos,
    const void* __restrict__ sin,
    void* __restrict__ grad_input,
    const int batch_size, const int seq_len, const int num_heads,
    const int x_stride_b, const int x_stride_s, const int x_stride_h,
    const int c_stride_b, const int c_stride_s, const int c_stride_h
) {
    constexpr int ELEMENTS_PER_THREAD = 8;
    constexpr int THREADS_PER_HEAD = HEAD_DIM / ELEMENTS_PER_THREAD;
    constexpr int SHUFFLE_MASK = (HEAD_DIM / 2) / ELEMENTS_PER_THREAD;
    constexpr int HEADS_PER_BLOCK = 256 / THREADS_PER_HEAD;
    int tid = threadIdx.x;
    int lane_id = tid % THREADS_PER_HEAD;
    int head_task_idx = tid / THREADS_PER_HEAD;
    
    int global_head_idx = blockIdx.x * HEADS_PER_BLOCK + head_task_idx;
    if (global_head_idx >= batch_size * seq_len * num_heads) return;

    int r = global_head_idx;
    int h_idx = r % num_heads; r /= num_heads;
    int s_idx = r % seq_len;   r /= seq_len;
    int b_idx = r;

    long long x_offset = (long long)b_idx * x_stride_b + 
                         (long long)s_idx * x_stride_s + 
                         (long long)h_idx * x_stride_h +
                         (lane_id * ELEMENTS_PER_THREAD);

    const __nv_bfloat16* g_ptr = reinterpret_cast<const __nv_bfloat16*>(grad_output);
    Packet8BF16 my_data = load_bf16_vec(g_ptr + x_offset);

    long long c_offset_base = (long long)b_idx * c_stride_b + 
                              (long long)s_idx * c_stride_s + 
                              (long long)h_idx * c_stride_h;
    long long cs_offset = c_offset_base + (lane_id * ELEMENTS_PER_THREAD);

    const float* cos_ptr = reinterpret_cast<const float*>(cos);
    const float* sin_ptr = reinterpret_cast<const float*>(sin);
    Packet4Float c0 = load_float_vec(cos_ptr + cs_offset);
    Packet4Float c1 = load_float_vec(cos_ptr + cs_offset + 4);
    Packet4Float s0 = load_float_vec(sin_ptr + cs_offset);
    Packet4Float s1 = load_float_vec(sin_ptr + cs_offset + 4);

    __nv_bfloat162* val_pairs = reinterpret_cast<__nv_bfloat162*>(&my_data.val);
    __nv_bfloat162 res_pairs[4];
    
    const float* cf0 = reinterpret_cast<const float*>(&c0.val);
    const float* cf1 = reinterpret_cast<const float*>(&c1.val);
    const float* sf0 = reinterpret_cast<const float*>(&s0.val);
    const float* sf1 = reinterpret_cast<const float*>(&s1.val);

    bool is_left = (lane_id < SHUFFLE_MASK);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        __nv_bfloat162 own = val_pairs[i];
        unsigned int own_as_int = *reinterpret_cast<unsigned int*>(&own);
        unsigned int pair_as_int = __shfl_xor_sync(0xffffffff,own_as_int,SHUFFLE_MASK);
        __nv_bfloat162 pair = *reinterpret_cast<__nv_bfloat162*>(&pair_as_int);

        float g_own_e = __bfloat162float(own.x); 
        float g_own_o = __bfloat162float(own.y);
        float g_pair_e = __bfloat162float(pair.x);
        float g_pair_o = __bfloat162float(pair.y);

        float ce, co, se, so;
        if (i < 2) {
            ce = cf0[2*i]; co = cf0[2*i+1]; se = sf0[2*i]; so = sf0[2*i+1];
        } else {
            int j = i - 2;
            ce = cf1[2*j]; co = cf1[2*j+1]; se = sf1[2*j]; so = sf1[2*j+1];
        }

        float out_e, out_o;
        
        if (is_left) {
            out_e = g_own_e * ce + g_pair_e * se;
            out_o = g_own_o * co + g_pair_o * so;
        } else {
            out_e = g_own_e * ce - g_pair_e * se;
            out_o = g_own_o * co - g_pair_o * so;
        }

        res_pairs[i] = __floats2bfloat162_rn(out_e, out_o);
    }

    Packet8BF16 out_vec;
    *reinterpret_cast<int4*>(&out_vec.val) = *reinterpret_cast<int4*>(res_pairs);
    
    __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(grad_input);
    store_bf16_vec(out_ptr + x_offset, out_vec);
}

torch::Tensor apply_rope_vision_backward_cuda(torch::Tensor grad, torch::Tensor cos, torch::Tensor sin) {
    if (!grad.is_contiguous()) grad = grad.contiguous();
    if (!cos.is_contiguous()) cos = cos.contiguous(); 
    if (!sin.is_contiguous()) sin = sin.contiguous();
    
    int embed_dim = grad.size(grad.dim()-1);
    int batch_size, seq_len, num_heads;
    int x_stride_b, x_stride_s, x_stride_h;
    int c_stride_b, c_stride_s, c_stride_h;

    if (grad.dim() == 3) {
        batch_size = 1;
        seq_len = grad.size(0);
        num_heads = grad.size(1);
        x_stride_b = 0; 
        x_stride_s = grad.stride(0); 
        x_stride_h = grad.stride(1);
    } else if (grad.dim() == 4) {
        batch_size = grad.size(0);
        seq_len = grad.size(1);
        num_heads = grad.size(2);
        x_stride_b = grad.stride(0); 
        x_stride_s = grad.stride(1); 
        x_stride_h = grad.stride(2);
    } else {
        TORCH_CHECK(false, "Grad must be 3D or 4D");
    }

    if (cos.dim() == 2) {
        // [S, D]
        c_stride_b = 0;
        c_stride_s = cos.stride(0);
        c_stride_h = 0;
    } else if (cos.dim() == 3) {
        // [B, S, D]
        c_stride_b = cos.stride(0);
        c_stride_s = cos.stride(1);
        c_stride_h = 0;
    } else if (cos.dim() == 4){
        if(cos.size(2) == 1){
            seq_len = grad.size(1);
            num_heads = grad.size(2);

            TORCH_CHECK(cos.size(1) == seq_len,"Cos seq_len mismatch (dim 1)");

            x_stride_b = grad.stride(0);
            x_stride_s = grad.stride(1);
            x_stride_h = grad.stride(2);

            c_stride_b = cos.stride(0);
            c_stride_s = cos.stride(1);
            c_stride_h = 0;
        } else if (cos.size(1) == 1){
            seq_len = grad.size(2);
            num_heads = grad.size(1);

            TORCH_CHECK(cos.size(2) == seq_len,"Cos seq_len mismatch (dim 1)");

            x_stride_b = grad.stride(0);
            x_stride_h = grad.stride(1);
            x_stride_s = grad.stride(2);

            c_stride_b = cos.stride(0);
            c_stride_h = 0;
            c_stride_s = cos.stride(2);
        }
    } else {
        TORCH_CHECK(false, "Cos/Sin must be 2D [S,D] or 3D [B,S,D] or 4D [B,S,1,D] or 4D [B,1,S,D]");
    }


    auto grad_input = torch::empty_like(grad);
    
    int total_heads = batch_size * seq_len * num_heads;
    
    if (embed_dim == 64) {
        constexpr int HEAD_DIM = 64;
        constexpr int THREADS_PER_HEAD = HEAD_DIM / 8; 
        constexpr int HEADS_PER_BLOCK = 256 / THREADS_PER_HEAD;
        int grid_size = (total_heads + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK;

        apply_rope_vision_backward_shfl_kernel<64><<<grid_size, 256>>>(
            grad.data_ptr(), cos.data_ptr(), sin.data_ptr(), grad_input.data_ptr(),
            batch_size, seq_len, num_heads,
            x_stride_b, x_stride_s, x_stride_h,
            c_stride_b, c_stride_s, c_stride_h
        );
    } else if(embed_dim == 128) {
        constexpr int HEAD_DIM = 128;
        constexpr int THREADS_PER_HEAD = HEAD_DIM / 8;
        constexpr int HEADS_PER_BLOCK = 256 / THREADS_PER_HEAD;
        int grid_size = (total_heads + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK;

        apply_rope_vision_backward_shfl_kernel<128><<<grid_size, 256>>>(
            grad.data_ptr(), cos.data_ptr(), sin.data_ptr(), grad_input.data_ptr(),
            batch_size, seq_len, num_heads,
            x_stride_b, x_stride_s, x_stride_h,
            c_stride_b, c_stride_s, c_stride_h
        );
    } else if(embed_dim == 256){
        constexpr int HEAD_DIM = 256;
        constexpr int THREADS_PER_HEAD = HEAD_DIM / 8;
        constexpr int HEADS_PER_BLOCK = 256 / THREADS_PER_HEAD;
        int grid_size = (total_heads + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK;

        apply_rope_vision_backward_shfl_kernel<256><<<grid_size, 256>>>(
            grad.data_ptr(), cos.data_ptr(), sin.data_ptr(), grad_input.data_ptr(),
            batch_size, seq_len, num_heads,
            x_stride_b, x_stride_s, x_stride_h,
            c_stride_b, c_stride_s, c_stride_h
        );
    }

    return grad_input;
}