//
// Created by Seeta on 2021/3/30.
//

#ifndef TENNIS_CONVERTER_OPTION_H
#define TENNIS_CONVERTER_OPTION_H



#include "module/module.h"
#include "utils/static.h"

namespace ts {

    class ConverterOption {
    public:
        using self = ConverterOption;

        virtual ~ConverterOption() = default;

        virtual std::shared_ptr<Module> convert(
                const ComputingDevice &device,
                const std::vector<Node> &outputs,
                const std::vector<Node> &inputs) const = 0;
    };

    TS_DEBUG_API const ConverterOption *QueryConverter(const std::string &device);

    TS_DEBUG_API void RegisterConverter(const std::string &device, const ConverterOption *option);
}

#define TS_REGISTER_CONVERTER_OPTION(device, option) \
namespace { \
    static option ts_serial_name(_ts_converter); \
    TS_STATIC_ACTION(ts::RegisterConverter, (device), &(ts_serial_name(_ts_converter))); \
}

#endif //TENNIS_CONVERTER_OPTION_H
