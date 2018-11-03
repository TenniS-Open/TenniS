//
// Created by kier on 2018/11/2.
//

#include "kernels/gpu/memory_gpu.h"
#include "utils/static.h"

#include "global/memory_device.h"

#include <cuda_runtime.h>
#include <iostream>

namespace ts {

    void *gpu_allocator(int id, size_t size, void *mem) {
        auto cuda_error = cudaSetDevice(id);
        if (cuda_error != cudaSuccess) {
            throw Exception("cudaSetDevice(" + std::to_string(id) + ") failed. error=" + std::to_string(cuda_error));
        }
        void *new_mem = nullptr;
        if (size == 0) {
            cudaFree(mem);
            return nullptr;
        } else if (mem != nullptr) {
            cudaFree(mem);
            cuda_error = cudaMalloc(&new_mem, size);
        } else {
            cuda_error = cudaMalloc(&new_mem, size);
        }
        if (new_mem == nullptr) throw OutOfMemoryException(MemoryDevice(CPU, id), size);
        if (cuda_error != cudaSuccess) {
            throw Exception("cudaMalloc(void**, " + std::to_string(size) + ") failed. error=" + std::to_string(cuda_error));
        }
        return new_mem;
    }

    void gpu2gpu_converter(int dst_id, void *dst, int src_id, const void *src, size_t size) {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }

    void cpu2gpu_converter(int dst_id, void *dst, int src_id, const void *src, size_t size) {
        auto cuda_error = cudaSetDevice(dst_id);
        if (cuda_error != cudaSuccess) {
            throw Exception("cudaSetDevice(" + std::to_string(dst_id) + ") failed. error=" + std::to_string(cuda_error));
        }
        cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }

    void gpu2cpu_converter(int dst_id, void *dst, int src_id, const void *src, size_t size) {
        auto cuda_error = cudaSetDevice(src_id);
        if (cuda_error != cudaSuccess) {
            throw Exception("cudaSetDevice(" + std::to_string(src_id) + ") failed. error=" + std::to_string(cuda_error));
        }
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
}

TS_STATIC_ACTION(ts::HardAllocator::Register, ts::GPU, ts::gpu_allocator)

TS_STATIC_ACTION(ts::HardConverter::Register, ts::GPU, ts::GPU, ts::gpu2gpu_converter)
TS_STATIC_ACTION(ts::HardConverter::Register, ts::GPU, ts::CPU, ts::cpu2gpu_converter)
TS_STATIC_ACTION(ts::HardConverter::Register, ts::CPU, ts::GPU, ts::gpu2cpu_converter)

TS_STATIC_ACTION(ts::RegisterMemoryDevice, ts::GPU, ts::GPU)