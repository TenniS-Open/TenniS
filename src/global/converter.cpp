//
// Created by lby on 2018/3/11.
//

#include "global/converter.h"
#include "utils/static.h"

#include <map>
#include <cassert>
#include <cstring>

namespace ts {
    static std::map<DeviceType, std::map<DeviceType, HardConverter>> map_dst_src_converter;

    HardConverter QueryConverter(DeviceType dst_device_type, DeviceType src_device_type) noexcept {
        auto dst_src_converter = map_dst_src_converter.find(dst_device_type);
        if (dst_src_converter != map_dst_src_converter.end()) {
            auto &map_src_converter = dst_src_converter->second;
            auto src_converter = map_src_converter.find(src_device_type);
            if (src_converter != map_src_converter.end()) {
                return src_converter->second;
            }
        }
        return HardConverter(nullptr);
    }

    void
    RegisterConverter(DeviceType dst_device_type, DeviceType src_device_type, const HardConverter &converter) noexcept {
        auto dst_src_converter = map_dst_src_converter.find(dst_device_type);
        if (dst_src_converter != map_dst_src_converter.end()) {
            auto &map_src_converter = dst_src_converter->second;
            map_src_converter.insert(std::make_pair(src_device_type, converter));
        } else {
            std::map<DeviceType, HardConverter> map_src_converter;
            map_src_converter.insert(std::make_pair(src_device_type, converter));
            map_dst_src_converter.insert(std::make_pair(dst_device_type, std::move(map_src_converter)));
        }
    }

    static void cpu_converter(const Device &dst_device, void *dst, const Device &src_device, const void *src, size_t size) {
        assert(dst_device.type() == ts::CPU);
        assert(src_device.type() == ts::CPU);
        std::memcpy(dst, src, size);
    }
}

TS_STATIC_ACTION(ts::RegisterConverter, ts::CPU, ts::CPU, ts::cpu_converter)
