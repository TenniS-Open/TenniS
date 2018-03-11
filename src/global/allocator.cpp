//
// Created by lby on 2018/3/11.
//

#include "global/allocator.h"

#include <map>
#include <cstdlib>

namespace ts {
    static std::map<DeviceType, HardAllocator> map_device_allocator;

    static void *cpu_allocator(size_t size, void *mem) {
        if (size == 0) {
            std::free(mem);
            return nullptr;
        } else if (mem != nullptr) {
            return std::realloc(mem, size);
        } else {
            return std::malloc(size);
        }
    }

    HardAllocator QueryAllocator(const Device &device) noexcept {
        auto device_allocator = map_device_allocator.find(device.type());
        if (device_allocator != map_device_allocator.end()) {
            return device_allocator->second;
        }
        return HardAllocator(nullptr);
    }

    void RegisterAllocator(const DeviceType &device_type, const HardAllocator &allocator) noexcept {
        map_device_allocator.insert(std::make_pair(device_type, allocator));
    }
}

TS_STATIC_ACTION(ts::RegisterAllocator, ts::CPU, ts::cpu_allocator)
