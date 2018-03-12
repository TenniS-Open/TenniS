//
// Created by lby on 2018/3/11.
//

#ifndef TENSORSTACK_GLOBAL_ALLOCATOR_H
#define TENSORSTACK_GLOBAL_ALLOCATOR_H

#include "device.h"
#include "utils/except.h"
#include <functional>

namespace ts {
    /**
     * Memory allocator type, allocate memory from specific device
     */
    using HardAllocator = std::function<void *(size_t, void *)>;

    /**
     * Query memory allocator
     * @param device querying device
     * @return allocator
     * @note supporting called by threads without calling @sa RegisterDeviceAllocator or @sa RegisterAllocator
     */
    HardAllocator QueryAllocator(const Device &device) TS_NOEXCEPT;

    /**
     * Register allocator for specific device type and id
     * @param device specific @sa Device
     * @param allocator setting allocator
     * @note only can be called before running
     */
	void RegisterDeviceAllocator(const Device &device, const HardAllocator &allocator) TS_NOEXCEPT;

    /**
     * Register allocator for specific device type
     * @param device_type specific @sa DeviceType
     * @param allocator setting allocator
     * @note only can be called before running
     */
	void RegisterAllocator(const DeviceType &device_type, const HardAllocator &allocator) TS_NOEXCEPT;
}

#endif //TENSORSTACK_GLOBAL_ALLOCATOR_H
