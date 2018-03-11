//
// Created by lby on 2018/3/11.
//

#ifndef TENSORSTACK_GLOBAL_ALLOCATOR_H
#define TENSORSTACK_GLOBAL_ALLOCATOR_H

#include "device.h"
#include <functional>

namespace ts {
    /**
     * Memory allocator type
     */
    using HardAllocator = std::function<void *(size_t, void *)>;

    HardAllocator QueryAllocator(const Device &device) noexcept;

    void RegisterAllocator(const Device &device, const HardAllocator &allocator) noexcept;

    void RegisterAllocator(const DeviceType &device_type, const HardAllocator &allocator) noexcept;
}

#define TS_REGISTER_ALLOCATOR(device, allocator) \
    namespace \
    { \
        struct _ts_local_register_allocator { _ts_local_register_allocator() noexcept { ts::RegisterAllocator((device), (allocator)); } }; \
        _ts_local_register_allocator _register_it; \
    }


#endif //TENSORSTACK_GLOBAL_ALLOCATOR_H
