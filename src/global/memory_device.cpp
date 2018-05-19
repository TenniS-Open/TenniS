//
// Created by seeta on 2018/5/19.
//

#include "global/memory_device.h"
#include <map>

namespace ts {

    static std::map<DeviceType, DeviceType> &MapDeviceMemory() {
        static std::map<DeviceType, DeviceType> map_device_memory;
        return map_device_memory;
    };

    DeviceType QueryMemoryDevice(const DeviceType &compute_device_type) {
        auto &map_device_memory = MapDeviceMemory();
        auto memory_device = map_device_memory.find(compute_device_type);
        if (memory_device != map_device_memory.end()) {
            return memory_device->second;
        }
        throw NoMemoryDeviceException(compute_device_type);
    }

    Device QueryMemoryDevice(const Device &compute_device) {
        return Device(QueryMemoryDevice(compute_device.type()), compute_device.id());
    }

    void RegisterMemoryDevice(const DeviceType &compute_device_type, const DeviceType &memory_device_type) TS_NOEXCEPT {
        auto &map_device_memory = MapDeviceMemory();
        map_device_memory.insert(std::make_pair(compute_device_type, memory_device_type));
    }
}