#ifndef TENNIS_GLOBAL_LOADER_FACTORY_H
#define TENNIS_GLOBAL_LOADER_FACTORY_H

#include <functional>
#include "module/module.h"

#include <map>
#include <set>

namespace ts {
    class TS_DEBUG_API ModuleLoader {
    public:
        using function = std::function<Module::shared(StreamReaderV2 &, const void*, int32_t, Module::SerializationFormat)>;

        static void Register(const std::string &module, const function &loader) TS_NOEXCEPT;

        static function Query(const std::string &module) TS_NOEXCEPT;

        static std::set<std::string> AllKeys() TS_NOEXCEPT;

        /**
         * No details for this API, so DO NOT call it
         */
        static void Clear();
    };
}

#endif //TENNIS_GLOBAL_LOADER_FACTORY_H
