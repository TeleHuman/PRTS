#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "qwenRoPE_kernel.cuh"


// Forward Kernel : Linear Load + Warp Shuffle
template<int HEAD_DIM>
__global__ void apply_rope_vision_forward_shfl_kernel(
    const void* __restrict__ x,
    const void* __restrict__ cos,
    const void* __restrict__ sin,
    void* __restrict__ x_out,
    const int batch_size, const int seq_len, const int num_heads,
    const int x_stride_b, const int x_stride_s, const int x_stride_h,
    const int c_stride_b, const int c_stride_s, const int c_stride_h
) {

    constexpr int ELEMENTS_PER_THREAD =8;
    constexpr int THREADS_PER_HEAD = HEAD_DIM / ELEMENTS_PER_THREAD;
    constexpr int SHUFFLE_MASK = (HEAD_DIM / 2) / ELEMENTS_PER_THREAD;

    constexpr int HEADS_PER_BLOCK = 256 /THREADS_PER_HEAD;

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
                         (lane_id * 8);

    const __nv_bfloat16* x_ptr = reinterpret_cast<const __nv_bfloat16*>(x);
    
    Packet8BF16 my_data = load_bf16_vec(x_ptr + x_offset);

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

        unsigned int past_as_int = __shfl_xor_sync(0xffffffff,own_as_int, SHUFFLE_MASK);

        __nv_bfloat162 pair = *reinterpret_cast<__nv_bfloat162*>(&past_as_int);

        float x_own_e = __bfloat162float(own.x);
        float x_own_o = __bfloat162float(own.y);
        float x_pair_e = __bfloat162float(pair.x);
        float x_pair_o = __bfloat162float(pair.y);

        float ce, co, se, so;
        if (i < 2) {
            ce = cf0[2*i]; co = cf0[2*i+1];
            se = sf0[2*i]; so = sf0[2*i+1];
        } else {
            int j = i - 2;
            ce = cf1[2*j]; co = cf1[2*j+1];
            se = sf1[2*j]; so = sf1[2*j+1];
        }

        float out_e, out_o;

        if (is_left) {
            out_e = x_own_e * ce - x_pair_e * se;
            out_o = x_own_o * co - x_pair_o * so;
        } else {
            out_e = x_own_e * ce + x_pair_e * se;
            out_o = x_own_o * co + x_pair_o * so;
        }

        res_pairs[i] = __floats2bfloat162_rn(out_e, out_o);
    }

    Packet8BF16 out_vec;
    *reinterpret_cast<int4*>(&out_vec.val) = *reinterpret_cast<int4*>(res_pairs);
    
    __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(x_out);
    store_bf16_vec(out_ptr + x_offset, out_vec);
}

torch::Tensor apply_rope_vision_forward_cuda(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
    if (!x.is_contiguous()) x = x.contiguous();
    if (!cos.is_contiguous()) cos = cos.contiguous(); 
    if (!sin.is_contiguous()) sin = sin.contiguous();

    int embed_dim = x.size(x.dim()-1);
    TORCH_CHECK(embed_dim == 64 || embed_dim == 128 || embed_dim == 256, "Head dim must be 64 or 128 or 256");

    int batch_size, seq_len, num_heads;
    int x_stride_b, x_stride_s, x_stride_h;
    int c_stride_b, c_stride_s, c_stride_h;

    if (x.dim() == 3) {
        batch_size = 1;
        seq_len = x.size(0);
        num_heads = x.size(1);
        
        x_stride_b = 0; 
        x_stride_s = x.stride(0);
        x_stride_h = x.stride(1);
    } else if (x.dim() == 4) {
        batch_size = x.size(0);
        seq_len = x.size(1);
        num_heads = x.size(2);
        
        x_stride_b = x.stride(0);
        x_stride_s = x.stride(1);
        x_stride_h = x.stride(2);
    } else {
        TORCH_CHECK(false, "Input x must be 3D [S,H,D] or 4D [B,S,H,D]");
    }

    if (cos.dim() == 2) {
        TORCH_CHECK(cos.size(0) == seq_len, "Cos seq_len mismatch");
        
        c_stride_b = 0; // Batch Broadcast
        c_stride_s = cos.stride(0);
        c_stride_h = 0; // Head Broadcast
        
    } else if (cos.dim() == 3) {
        TORCH_CHECK(cos.size(0) == batch_size, "Cos batch_size mismatch");
        TORCH_CHECK(cos.size(1) == seq_len, "Cos seq_len mismatch");
        
        c_stride_b = cos.stride(0);
        c_stride_s = cos.stride(1);
        c_stride_h = 0; // Head Broadcast
        
    } else if (cos.dim() == 4){
        if(cos.size(2) == 1){
            seq_len = x.size(1);
            num_heads = x.size(2);

            TORCH_CHECK(cos.size(1) == seq_len,"Cos seq_len mismatch (dim 1)");

            x_stride_b = x.stride(0);
            x_stride_s = x.stride(1);
            x_stride_h = x.stride(2);

            c_stride_b = cos.stride(0);
            c_stride_s = cos.stride(1);
            c_stride_h = 0;
        } else if (cos.size(1) == 1){
            seq_len = x.size(2);
            num_heads = x.size(1);

            TORCH_CHECK(cos.size(2) == seq_len,"Cos seq_len mismatch (dim 1)");

            x_stride_b = x.stride(0);
            x_stride_h = x.stride(1);
            x_stride_s = x.stride(2);

            c_stride_b = cos.stride(0);
            c_stride_h = 0;
            c_stride_s = cos.stride(2);
        }
    } else {
        TORCH_CHECK(false, "Cos/Sin must be 2D [S,D] or 3D [B,S,D] or 4D [B,S,1,D] or 4D [B,1,S,D]");
    }

    auto x_out = torch::empty_like(x);

    int total_heads = batch_size * seq_len * num_heads;
    if(embed_dim == 64){
        constexpr int HEAD_DIM = 64;
        constexpr int THREADS_PER_HEAD = HEAD_DIM / 8;
        constexpr int HEADS_PER_BLOCK = 256 / THREADS_PER_HEAD;
        int grid_size = (total_heads + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK;

        apply_rope_vision_forward_shfl_kernel<64><<<grid_size, 256>>>(
        x.data_ptr(), cos.data_ptr(), sin.data_ptr(), x_out.data_ptr(),
        batch_size, seq_len, num_heads,
        x_stride_b, x_stride_s, x_stride_h,
        c_stride_b, c_stride_s, c_stride_h
        );
    }else if(embed_dim == 128){
        constexpr int HEAD_DIM = 128;
        constexpr int THREADS_PER_HEAD = HEAD_DIM / 8;
        constexpr int HEADS_PER_BLOCK = 256 / THREADS_PER_HEAD;
        int grid_size = (total_heads + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK;

        apply_rope_vision_forward_shfl_kernel<128><<<grid_size, 256>>>(
        x.data_ptr(), cos.data_ptr(), sin.data_ptr(), x_out.data_ptr(),
        batch_size, seq_len, num_heads,
        x_stride_b, x_stride_s, x_stride_h,
        c_stride_b, c_stride_s, c_stride_h
        );
    }else{
        constexpr int HEAD_DIM = 256;
        constexpr int THREADS_PER_HEAD = HEAD_DIM / 8;
        constexpr int HEADS_PER_BLOCK = 256 / THREADS_PER_HEAD;
        int grid_size = (total_heads + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK;

        apply_rope_vision_forward_shfl_kernel<256><<<grid_size, 256>>>(
        x.data_ptr(), cos.data_ptr(), sin.data_ptr(), x_out.data_ptr(),
        batch_size, seq_len, num_heads,
        x_stride_b, x_stride_s, x_stride_h,
        c_stride_b, c_stride_s, c_stride_h
        );        
    }

    return x_out;
}