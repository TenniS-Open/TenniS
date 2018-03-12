//
// Created by lby on 2018/3/11.
//

#ifndef TENSORSTACK_MEM_CONTROLLER_H
#define TENSORSTACK_MEM_CONTROLLER_H

#include "memory.h"

namespace ts {
    /**
     * MemoryController: Malloc memory and control them
     */
    class MemoryController {
    public:
        /**
         * alloc memory with size
         * @param size memory size (bytes)
         * @return allocated memory
         */
        virtual Memory alloc(size_t size) = 0;
    };

    class BaseMemoryController : public MemoryController {
    public:
        explicit BaseMemoryController(const Device &device);

        Memory alloc(size_t size) override;

    private:
        Device m_device;
        HardAllocator m_allocator;
    };
}


#endif //TENSORSTACK_MEM_CONTROLLER_H
