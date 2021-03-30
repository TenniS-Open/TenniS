//
// Created by Seeta on 2021/3/30.
//

#include "compiler/option/converter_option.h"

namespace ts {
    static std::unordered_map<std::string, const ConverterOption *> &GetMap() {
        static std::unordered_map<std::string, const ConverterOption *> map;
        return map;
    }

    TS_DEBUG_API const ConverterOption *QueryConverter(const std::string &device) {
        auto &map = GetMap();
        auto it = map.find(device);
        if (it == map.end()) return nullptr;
        return it->second;
    }

    TS_DEBUG_API void RegisterConverter(const std::string &device, const ConverterOption *option) {
        GetMap()[device] = option;
    }
}
