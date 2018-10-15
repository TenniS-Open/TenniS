//
// Created by seeta on 2018/5/19.
//

#include "global/memory_device.h"
#include <map>

namespace ts {

    static std::map<DeviceType, DeviceType> &MapMemoryDevice() {
        static std::map<DeviceType, DeviceType> map_memory_device;
        return map_memory_device;
    };

    DeviceType QueryMemoryDevice(const DeviceType &compute_device_type) {
        auto &map_memory_device = MapMemoryDevice();
        auto memory_device = map_memory_device.find(compute_device_type);
        if (memory_device != map_memory_device.end()) {
            return memory_device->second;
        }
        throw NoMemoryDeviceException(compute_device_type);
    }

    MemoryDevice QueryMemoryDevice(const Device &compute_device) {
        return MemoryDevice(QueryMemoryDevice(compute_device.type()), compute_device.id());
    }

    void RegisterMemoryDevice(const DeviceType &compute_device_type, const DeviceType &memory_device_type) TS_NOEXCEPT {
        auto &map_memory_device = MapMemoryDevice();
        map_memory_device.insert(std::make_pair(compute_device_type, memory_device_type));
    }

    void ClearRegisteredMemoryDevice() {
        auto &map_memory_device = MapMemoryDevice();
        map_memory_device.clear();
    }
}