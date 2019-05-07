//
// Created by kier on 19-5-7.
//

#ifndef TENSORSTACK_API_CPP_PROGORAM_H
#define TENSORSTACK_API_CPP_PROGORAM_H

#include "../program.h"

#include "except.h"
#include "device.h"
#include "module.h"

#include <string>

namespace ts {
    namespace api {
        class Workbench;
        class Program {
        public:
            using self = Program;
            using raw = ts_Program;

            using shared = std::shared_ptr<self>;
            using shared_raw = std::shared_ptr<raw>;

            Program(const self &) = default;

            Program &operator=(const self &) = default;

            raw *get_raw() const { return m_impl.get(); }

            Program() = default;

            static Program Compile(const Module &module, const Device &device) {
                return Compile(module.get_raw(), device.get_raw());
            }

            static Program Compile(const ts_Module *module, const Device &device) {
                return Compile(module, device.get_raw());
            }

            static Program Compile(const Module &module, const ts_Device *device) {
                return Compile(module.get_raw(), device);
            }

            static Program Compile(const ts_Module *module, const ts_Device *device) {
                Program loaded(ts_Program_Compile(module, device));
                TS_API_AUTO_CHECK(loaded.m_impl != nullptr);
                return std::move(loaded);
            }

            Program clone() const {
                Program dolly(ts_Program_clone(m_impl.get()));
                TS_API_AUTO_CHECK(dolly.m_impl != nullptr);
                return std::move(dolly);
            }

        private:
            friend class Workbench;

            Program(raw *ptr) : m_impl(pack(ptr)) {}

            static shared_raw pack(raw *ptr) { return shared_raw(ptr, ts_free_Program); }

            shared_raw m_impl;
        };
    }
}

#endif //TENSORSTACK_API_CPP_PROGORAM_H
