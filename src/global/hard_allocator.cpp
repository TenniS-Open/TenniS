//
// Created by lby on 2018/3/11.
//

#include "global/hard_allocator.h"
#include "utils/static.h"

#include <map>
#include <cstdlib>
#include <iostream>

namespace ts {
    static std::map<DeviceType, HardAllocator> &MapDeviceAllocator() {
        static std::map<DeviceType, HardAllocator> map_device_allocator;
        return map_device_allocator;
    };

	HardAllocator QueryAllocator(const MemoryDevice &device) TS_NOEXCEPT{
        auto &map_device_allocator = MapDeviceAllocator();
        auto device_allocator = map_device_allocator.find(device.type());
        if (device_allocator != map_device_allocator.end()) {
            return device_allocator->second;
        }
        return HardAllocator(nullptr);
    }

    void RegisterAllocator(const DeviceType &device_type, const HardAllocator &allocator) TS_NOEXCEPT {
        auto &map_device_allocator = MapDeviceAllocator();
        map_device_allocator.insert(std::make_pair(device_type, allocator));
    }

    void ClearRegisteredAllocator() {
        auto &map_device_allocator = MapDeviceAllocator();
        map_device_allocator.clear();
    }
}
