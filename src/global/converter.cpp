//
// Created by lby on 2018/3/11.
//

#include "global/converter.h"
#include "utils/static.h"

#include <map>

#include <iostream>

namespace ts {
    static std::map<DeviceType, std::map<DeviceType, HardConverter>> &MapDstSrcConverter() {
        static std::map<DeviceType, std::map<DeviceType, HardConverter>> map_dst_src_converter;
        return map_dst_src_converter;
    };

	HardConverter QueryConverter(DeviceType dst_device_type, DeviceType src_device_type) TS_NOEXCEPT{
        auto &map_dst_src_converter = MapDstSrcConverter();
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
	RegisterConverter(DeviceType dst_device_type, DeviceType src_device_type, const HardConverter &converter) TS_NOEXCEPT{
        auto &map_dst_src_converter = MapDstSrcConverter();
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
}
