//
// Created by lby on 2018/3/12.
//

#include "global/register.h"

#include "utils/static.h"

#include "global/allocator.h"
#include "global/converter.h"

#include <cassert>
#include <cstring>

namespace ts {
    static void *cpu_allocator(int id, size_t size, void *mem) {
        TS_UNUSED(id);
        if (size == 0) {
            std::free(mem);
            return nullptr;
        } else if (mem != nullptr) {
            return std::realloc(mem, size);
        } else {
            return std::malloc(size);
        }
    }

    static void
    cpu_converter(const Device &dst_device, void *dst, const Device &src_device, const void *src, size_t size) {
        assert(dst_device.type() == ts::CPU);
        assert(src_device.type() == ts::CPU);
        std::memcpy(dst, src, size);
    }

    void RegisterGlobal() {
    }
}

TS_STATIC_ACTION(ts::RegisterAllocator, ts::CPU, ts::cpu_allocator)

TS_STATIC_ACTION(ts::RegisterConverter, ts::CPU, ts::CPU, ts::cpu_converter)
