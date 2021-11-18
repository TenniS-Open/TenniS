#include "kernels/cpu/memory_cpu.h"

#include "utils/static.h"

#include "global/hard_allocator.h"
#include "global/hard_converter.h"
#include "global/memory_device.h"

#include "utils/assert.h"

#include <cstring>

namespace ts {
    static void *map2align(void *data, size_t stride) {
        if (!data) return nullptr;
        auto align = (char *)data + (stride - (size_t) data % stride);
        auto *shift = align - 1;
        *shift = (char)(stride - (size_t) data % stride);
        return align;
    }

    static void *map2origin(void *data) {
        if (!data) return nullptr;
        auto *shift = (char *)data - 1;
        return (char *)data - (int)*shift;
    }

    void align_free(void *data) {
        std::free(map2origin(data));
    }

    void *align_malloc(size_t size, size_t stride=16, size_t page=16) {
        return map2align(std::malloc(size + stride + page), stride);
    }

    void *align_realloc(void *data, size_t size, size_t stride=16, size_t page=16) {
        return map2align(std::realloc(map2origin(data), size + stride + page), stride);
    }

    void *cpu_allocator(int id, size_t new_size, void *mem, size_t mem_size) {
        if (new_size == 0 && mem == nullptr) return nullptr;
        void *new_mem = nullptr;
        if (new_size == 0) {
            align_free(mem);
            return nullptr;
        } else if (mem != nullptr) {
            if (mem_size) {
                new_mem = align_realloc(mem, new_size);
            } else {
                align_free(mem);
                new_mem = align_malloc(new_size);
            }
        } else {
            new_mem = align_malloc(new_size);
        }
        if (new_mem == nullptr) throw OutOfMemoryException(MemoryDevice(CPU, id), new_size);
        return new_mem;
    }

    void
    cpu_converter(int dst_id, void *dst, int src_id, const void *src, size_t size) {
        TS_UNUSED(dst_id);
        TS_UNUSED(src_id);
        std::memcpy(dst, src, size);
    }
}

TS_STATIC_ACTION(ts::HardAllocator::Register, ts::CPU, ts::cpu_allocator)

TS_STATIC_ACTION(ts::HardConverter::Register, ts::CPU, ts::CPU, ts::cpu_converter)

TS_STATIC_ACTION(ts::ComputingMemory::Register, ts::CPU, ts::CPU)