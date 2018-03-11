//
// Created by lby on 2018/3/11.
//

#ifndef TENSORSTACK_GLOBAL_CONVERTER_H
#define TENSORSTACK_GLOBAL_CONVERTER_H


#include "device.h"
#include <functional>

namespace ts {
    /**
     * HardConverter, convert memory between devices
     */
    using HardConverter = std::function<void(const Device &, void *, const Device &, const void *, size_t)>;

    /**
     * Query memory converter
     * @param dst_device_type querying dst device
     * @param src_device_type querying src device
     * @return converter
     */
    HardConverter QueryConverter(DeviceType dst_device_type, DeviceType src_device_type) noexcept;

    /**
     * Register converter
     * @param dst_device_type registering dst device
     * @param src_device_type registering src device
     * @param converter registering converter
     */
    void RegisterConverter(DeviceType dst_device_type, DeviceType src_device_type, const HardConverter &converter) noexcept;
}

#endif //TENSORSTACK_GLOBAL_CONVERTER_H
