//
// Created by lby on 2018/3/11.
//

#ifndef TENSORSTACK_MEM_MEMORY_H
#define TENSORSTACK_MEM_MEMORY_H

#include "global/device.h"
#include "hard_memory.h"

#include <memory>

namespace ts {

    /**
     * Memory, directly memory on specific device
     */
    class Memory {
    public:
        using self = Memory;    ///< self class

        /**
         * Initialize Memory
         * @param device running device
         * @param size sizeof the memory block
         * @param hard ready memory
         * @param shift shift from start pointer
         */
        Memory(const Device &device, size_t size, const std::shared_ptr<HardMemory> &hard, size_t shift = 0);

        /**
         * Initialize Memory
         * @param device running device
         * @param size sizeof the memory block
         * @param hard ready memory
         * @param shift shift from start pointer
         */
        Memory(const Device &device, size_t size, std::shared_ptr<HardMemory> &&hard, size_t shift = 0);

        /**
         * Initialize Memory
         * @param device running device
         * @param size sizeof this memory block
         */
        Memory(const Device &device, size_t size);

        /**
         * Initialize Memory, with cpu zero memory
         * @param size sizeof the memory block
         */
        explicit Memory(size_t size);

    private:
        Device m_device;    ///< Running device
        size_t m_size;  ///< sizeof this memory block
        std::shared_ptr<HardMemory> m_hard; ///< hardware memory
        size_t m_shift; ///< shift from start pointer
    };

}


#endif //TENSORSTACK_MEM_MEMORY_H
