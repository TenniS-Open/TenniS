#ifndef TENNIS_MODULEV2_H
#define TENNIS_MODULEV2_H

#include "module/module.h"

namespace ts {
    class TS_DEBUG_API ModuleV2 : public Module {
    public:
        using self = ModuleV2;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        static ModuleV2::shared Load(StreamReaderV2 &stream, const void *buffer, int32_t buffer_size,
                                     ModuleV2::SerializationFormat format);
        static ModuleV2::shared Load(const std::string &filename, const void *buffer, int32_t buffer_size,
                                     ModuleV2::SerializationFormat format = BINARY);
    };

    class FormatMismatchException : public Exception {
    public:
        FormatMismatchException() : Exception() {}
        explicit FormatMismatchException(const std::string &message) : Exception(message) {}
    };
}

#endif //TENNIS_MODULEV2_H
