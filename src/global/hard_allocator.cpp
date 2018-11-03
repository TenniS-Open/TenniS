//
// Created by lby on 2018/3/11.
//

#include "global/hard_allocator.h"
#include "utils/static.h"

#include <map>
#include <cstdlib>
#include <iostream>

namespace ts {
    static std::map<DeviceType, HardAllocator::function> &MapDeviceAllocator() {
        static std::map<DeviceType, HardAllocator::function> map_device_allocator;
        return map_device_allocator;
    };

    HardAllocator::function HardAllocator::Query(const DeviceType &device_type) TS_NOEXCEPT{
        auto &map_device_allocator = MapDeviceAllocator();
        auto device_allocator = map_device_allocator.find(device_type);
        if (device_allocator != map_device_allocator.end()) {
            return device_allocator->second;
        }
        return HardAllocator::function(nullptr);
    }

    void HardAllocator::Register(const DeviceType &device_type, const function &allocator) TS_NOEXCEPT {
        auto &map_device_allocator = MapDeviceAllocator();
        map_device_allocator.insert(std::make_pair(device_type, allocator));
    }

    void HardAllocator::Clear() {
        auto &map_device_allocator = MapDeviceAllocator();
        map_device_allocator.clear();
    }
}
