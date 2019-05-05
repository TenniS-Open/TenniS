//
// Created by kier on 2019/3/16.
//

#ifndef TENSORSTACK_API_CPP_MODULE_H
#define TENSORSTACK_API_CPP_MODULE_H

#include "../module.h"

#include "except.h"
#include "device.h"
#include "stream.h"

#include <string>

namespace ts {
    namespace api {
        class SerializationFormat {
        public:
            using self = SerializationFormat;

            SerializationFormat() : self(TS_BINARY) {}

            SerializationFormat(ts_SerializationFormat dtype) : raw(dtype) {}

            operator ts_SerializationFormat() const { return raw; }

            ts_SerializationFormat raw;
        };

        static const SerializationFormat BINARY = TS_BINARY;
        static const SerializationFormat TEXT = TS_TEXT;

        class Module {
        public:
            using self = Module;
            using raw = ts_Module;

            using shared = std::shared_ptr<self>;
            using shared_raw = std::shared_ptr<raw>;

            Module(const self &) = default;

            Module &operator=(const self &) = default;

            raw *get_raw() const { return m_impl.get(); }

            Module() = default;

            static Module Load(const std::string &path, SerializationFormat format = TS_BINARY) {
                Module loaded(ts_Module_Load(path.c_str(), ts_SerializationFormat(format)));
                TS_API_AUTO_CHECK(loaded.m_impl != nullptr);
                return std::move(loaded);
            }

            static Module Load(StreamReader &stream, SerializationFormat format = TS_BINARY) {
                Module loaded(ts_Module_LoadFromStream(&stream, StreamReader::C, ts_SerializationFormat(format)));
                TS_API_AUTO_CHECK(loaded.m_impl != nullptr);
                return std::move(loaded);
            }

        private:
            Module(raw *ptr) : m_impl(pack(ptr)) {}

            static shared_raw pack(raw *ptr) { return shared_raw(ptr, ts_free_Module); }

            shared_raw m_impl;
        };
    }
}

#endif //TENSORSTACK_API_CPP_MODULE_H
