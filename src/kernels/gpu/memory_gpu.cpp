//
// Created by kier on 2018/11/2.
//

#include "kernels/gpu/memory_gpu.h"
#include "utils/static.h"
#include "utils/assert.h"

#include "global/memory_device.h"

#include <cuda_runtime.h>
#include <iostream>

namespace ts {
    class CUDAException : public Exception {
    public:
        CUDAException() : Exception() {}
        explicit CUDAException(const std::string &api, cudaError_t cuda_error)
                : Exception(CUDAExceptionMessage(api, cuda_error)) {}

        cudaError_t cudaError() const { return m_cudaError; }
    private:
        static std::string CUDAExceptionMessage(const std::string &api, cudaError_t cuda_error) {
            std::ostringstream oss;
            oss << "Call " << api << " failed. error=" << cuda_error << ".";
            return oss.str();
        }
        cudaError_t m_cudaError = cudaSuccess;
    };

    void *gpu_allocator(int id, size_t size, void *mem) {
        auto cuda_error = cudaSetDevice(id);
        if (cuda_error != cudaSuccess) {
            TS_LOG_ERROR << "cudaSetDevice(" << id << ") failed. error=" << cuda_error << eject;
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
            TS_LOG_ERROR << "cudaMalloc(" << &new_mem << " " << size << ") failed. error=" << cuda_error << eject;
        }
        return new_mem;
    }

    void gpu2gpu_converter(int dst_id, void *dst, int src_id, const void *src, size_t size) {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }

    void cpu2gpu_converter(int dst_id, void *dst, int src_id, const void *src, size_t size) {
        auto cuda_error = cudaSetDevice(dst_id);
        if (cuda_error != cudaSuccess) {
            TS_LOG_ERROR << "cudaSetDevice(" << dst_id << ") failed. error=" << cuda_error << eject;
        }
        cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }

    void gpu2cpu_converter(int dst_id, void *dst, int src_id, const void *src, size_t size) {
        auto cuda_error = cudaSetDevice(src_id);
        if (cuda_error != cudaSuccess) {
            TS_LOG_ERROR << "cudaSetDevice(" << src_id << ") failed. error=" << cuda_error << eject;
        }
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
}

TS_STATIC_ACTION(ts::HardAllocator::Register, ts::GPU, ts::gpu_allocator)

TS_STATIC_ACTION(ts::HardConverter::Register, ts::GPU, ts::GPU, ts::gpu2gpu_converter)
TS_STATIC_ACTION(ts::HardConverter::Register, ts::GPU, ts::CPU, ts::cpu2gpu_converter)
TS_STATIC_ACTION(ts::HardConverter::Register, ts::CPU, ts::GPU, ts::gpu2cpu_converter)

TS_STATIC_ACTION(ts::ComputingMemory::Register, ts::GPU, ts::GPU)