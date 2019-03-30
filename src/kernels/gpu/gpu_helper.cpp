//
// Created by kier on 19-3-30.
//

#include "kernels/gpu/gpu_helper.h"

namespace ts {
    namespace gpu {
        static inline std::shared_ptr<char> cpu_blocked(const std::vector<CpuBlock> &cpu) {
            int shift = 0;
            for (auto &pair : cpu) {
                shift += pair.second;
            }
            std::shared_ptr<char> cpu_memory(new char[shift], std::default_delete<char[]>());
            shift = 0;
            for (auto &pair : cpu) {
                std::memcpy(cpu_memory.get() + shift, pair.first, size_t(pair.second));
                shift += pair.second;
            }
            return std::move(cpu_memory);
        }
        SyncMemory convert_block_to_gpu(
                SyncMemoryController::shared controller,
                const MemoryDevice &device,
                const std::vector<CpuBlock> &cpu,
                const std::vector<void **> &gpu) {
            int shift = 0;
            for (auto &pair : cpu) {
                shift += pair.second;
            }
            std::shared_ptr<char> cpu_memory(new char[shift], std::default_delete<char[]>());
            auto cpu_data = cpu_memory.get();
            auto gpu_memory = controller->alloc(device, size_t(shift));
            auto gpu_data = gpu_memory.data<char>();

            shift = 0;
            for (size_t i = 0; i < cpu.size(); ++i) {
                auto &pair = cpu[i];
                std::memcpy(cpu_data + shift, pair.first, size_t(pair.second));
                *gpu[i] = gpu_data + shift;
                shift += pair.second;
            }

            memcpy(gpu_data, device, shift, cpu_data, MemoryDevice(CPU), shift);

            return std::move(gpu_memory);
        }
    }
}
