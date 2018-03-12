//
// Created by lby on 2018/3/11.
//

#include <cassert>
#include "mem/controller.h"

namespace ts {

    BaseMemoryController::BaseMemoryController(const Device &device)
            : m_device(device) {
        m_allocator = QueryAllocator(device);
        assert(m_allocator != nullptr);
    }

    Memory BaseMemoryController::alloc(size_t size) {
        return Memory(std::make_shared<HardMemory>(m_device, m_allocator, size), size);
    }
}