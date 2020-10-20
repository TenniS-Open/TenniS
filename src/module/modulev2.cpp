#include <module/header.h>
#include <global/module_loader_factory.h>
#include <module/io/fstream.h>
#include "module/modulev2.h"

using namespace ts;

ModuleV2::shared ModuleV2::Load(StreamReaderV2 &stream, const void *buffer, int32_t buffer_size,
                                ModuleV2::SerializationFormat format) {

    auto module_loaders = ModuleLoader::AllKeys();
    for (auto &module_loader : module_loaders) {
        auto loader = ModuleLoader::Query(module_loader);
        try {
            auto m = loader(stream, buffer, buffer_size, format);
            return m;
        } catch (const FormatMismatchException &e) {
            stream.rewind();
            continue;
        }
    }
    throw FormatMismatchException("Format not recognized");
}

ModuleV2::shared ModuleV2::Load(const std::string &filename, const void *buffer, int32_t buffer_size,
                                ModuleV2::SerializationFormat format) {
    FileStreamReaderV2 stream(filename);
    // all SerializationFormat checked here
    TS_CHECK(format == BINARY);
    return Load(stream, buffer, buffer_size, format);
}

