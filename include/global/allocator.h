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
     * @see HardAllocatorDeclaration
     */
    using HardAllocator = std::function<void *(int, size_t, void *)>;

    /**
     * Example of HardAllocator
     * @param id the allocating device id
     * @param size the new size of memory
     * @param mem the older memory
     * @return a pointer to new memory
     * @note if size == 0: free(mem),
     *        else if mem == nullptr: return malloc(size)
     *        else: return realloc(mem, size)
     */
    void *HardAllocatorDeclaration(int id, size_t size, void *mem);

    /**
     * Query memory allocator
     * @param device querying device
     * @return allocator
     * @note supporting called by threads without calling @sa RegisterDeviceAllocator or @sa RegisterAllocator
     * @note the query device should be memory device, you may call @sa QueryMemoryDevice to get memory device by compute device
     */
    HardAllocator QueryAllocator(const Device &device) TS_NOEXCEPT;

    /**
     * Register allocator for specific device type
     * @param device_type specific @sa DeviceType
     * @param allocator setting allocator
     * @note only can be called before running
     */
    void RegisterAllocator(const DeviceType &device_type, const HardAllocator &allocator) TS_NOEXCEPT;
}

#endif //TENSORSTACK_GLOBAL_ALLOCATOR_H
