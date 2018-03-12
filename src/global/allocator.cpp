//
// Created by lby on 2018/3/11.
//

#include "global/allocator.h"
#include "utils/static.h"

#include <map>
#include <cstdlib>
#include <iostream>

namespace ts {
    static std::map<DeviceType, HardAllocator> &MapDeviceAllocator() {
        static std::map<DeviceType, HardAllocator> map_device_allocator;
        return map_device_allocator;
    };

    static std::map<DeviceType, std::map<int, HardAllocator>> &MapDeviceIDAllocator() {
        static std::map<DeviceType, std::map<int, HardAllocator>> map_device_id_allocator;
        return map_device_id_allocator;

    };

	HardAllocator QueryAllocator(const Device &device) TS_NOEXCEPT{
        auto &map_device_allocator = MapDeviceAllocator();
        auto device_allocator = map_device_allocator.find(device.type());
        if (device_allocator != map_device_allocator.end()) {
            return device_allocator->second;
        }
        auto &map_device_id_allocator = MapDeviceIDAllocator();
        auto device_id_allocator = map_device_id_allocator.find(device.type());
        if (device_id_allocator != map_device_id_allocator.end()) {
            auto &map_id_allocator = device_id_allocator->second;
            auto id_allocator = map_id_allocator.find(device.id());
            if (id_allocator != map_id_allocator.end()) {
                return id_allocator->second;
            }
        }
        return HardAllocator(nullptr);
    }

    void RegisterDeviceAllocator(const Device &device, const HardAllocator &allocator) TS_NOEXCEPT {
        auto &map_device_id_allocator = MapDeviceIDAllocator();
        auto device_id_allocator = map_device_id_allocator.find(device.type());
        if (device_id_allocator != map_device_id_allocator.end()) {
            auto &map_id_allocator = device_id_allocator->second;
            map_id_allocator.insert(std::make_pair(device.id(), allocator));
        } else {
            std::map<int, HardAllocator> map_id_allocator;
            map_id_allocator.insert(std::make_pair(device.id(), allocator));
            map_device_id_allocator.insert(std::make_pair(device.type(), std::move(map_id_allocator)));
        }
    }

    void RegisterAllocator(const DeviceType &device_type, const HardAllocator &allocator) TS_NOEXCEPT {
        auto &map_device_allocator = MapDeviceAllocator();
        map_device_allocator.insert(std::make_pair(device_type, allocator));
    }
}
