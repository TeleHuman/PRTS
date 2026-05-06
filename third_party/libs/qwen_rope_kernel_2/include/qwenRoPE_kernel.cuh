#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>


// 128-bit 包
struct Packet8BF16 { int4 val; };
struct Packet4Float { float4 val; };

__device__ __forceinline__ Packet8BF16 load_bf16_vec(const void* ptr) {
    Packet8BF16 ret; ret.val = *reinterpret_cast<const int4*>(ptr); return ret;
}
__device__ __forceinline__ void store_bf16_vec(void* ptr, Packet8BF16 data) {
    *reinterpret_cast<int4*>(ptr) = data.val;
}
__device__ __forceinline__ Packet4Float load_float_vec(const void* ptr) {
    Packet4Float ret; ret.val = *reinterpret_cast<const float4*>(ptr); return ret;
}
inline int get_broadcast_stride(const torch::Tensor& t, int dim_from_end, int expected_size) {
    int dim = t.dim();
    int target_dim = dim - 1 - dim_from_end;
    
    if (target_dim < 0) {
        return 0;
    }
    
    int size = t.size(target_dim);
    if (size == 1) {
        return 0;
    }
    
    TORCH_CHECK(size == expected_size, 
        "Broadcasting mismatch: Expected size ", expected_size, 
        " at dim -", dim_from_end + 1, 
        " but got ", size);
        
    return t.stride(target_dim);
}