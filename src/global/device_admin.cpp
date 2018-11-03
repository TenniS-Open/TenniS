//
// Created by kier on 2018/11/2.
//

#include "global/device_admin.h"

#include <map>
#include <cstdlib>
#include <iostream>

namespace ts {
    static std::map<DeviceType, DeviceAdmin> &MapDeviceTypeAdmin() {
        static std::map<DeviceType, DeviceAdmin> map_device_type_admin;
        return map_device_type_admin;
    };

    DeviceAdmin QueryDeviceAdmin(const DeviceType &device_type) TS_NOEXCEPT{
        auto &map_device_type_admin = MapDeviceTypeAdmin();
        auto device_type_admin = map_device_type_admin.find(device_type);
        if (device_type_admin != map_device_type_admin.end()) {
            return device_type_admin->second;
        }
        return DeviceAdmin(nullptr);
    }

    void RegisterDeviceAdmin(const DeviceType &device_type, const DeviceAdmin &device_admin) TS_NOEXCEPT {
        auto &map_device_type_admin = MapDeviceTypeAdmin();
        map_device_type_admin.insert(std::make_pair(device_type, device_admin));
    }

    void ClearRegisteredDeviceAdmin() {
        auto &map_device_type_admin = MapDeviceTypeAdmin();
        map_device_type_admin.clear();
    }
}
