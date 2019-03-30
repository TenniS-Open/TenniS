//
// Created by kier on 19-3-30.
//

#ifndef TENSORSTACK_GPU_HELPER_H
#define TENSORSTACK_GPU_HELPER_H


#include <core/sync/sync_memory.h>
#include <core/sync/sync_controller.h>

namespace ts {
    namespace gpu {
        using CpuBlock = std::pair<void *, int>;
        /**
         * convert cpu memory to gpu, by using single gpu memory block
         * @param [in] controller controller to memory alloc
         * @param [in] device device contain memory
         * @param [in] cpu list of cpu memory
         * @param [out] gpu pointer containing converted gpu memory pointer
         * @return the memory contain all `gpu` memory
         */
        SyncMemory convert_block_to_gpu(
                SyncMemoryController::shared controller,
                const MemoryDevice &device,
                const std::vector<CpuBlock> &cpu,
                const std::vector<void **> &gpu);
    }
}


#endif //TENSORSTACK_GPU_HELPER_H
