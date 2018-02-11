//
// Created by lby on 2018/2/11.
//

#include "mem/hard_memory.h"

#include <cstdlib>
#include <utility>

namespace ts {

    static void *cpu_allocator(size_t size, void *mem) {
        if (size == 0) {
            std::free(mem);
            return nullptr;
        } else if (mem != nullptr) {
            return std::realloc(mem, size);
        } else {
            return std::malloc(size);
        }
    }

    static HardAllocator *default_allocator(Device device) {
        switch (device.type) {
            case CPU:
                return cpu_allocator;
            case GPU:
                throw "Unsupported device.";
            default:
                throw "Unknown device type.";
        }
    }

    HardMemory::HardMemory(const Device &device)
            : m_device(device) {
        m_allocator = default_allocator(device);
    }

    HardMemory::~HardMemory() {
        if (m_allocator) m_allocator(0, m_data);
    }

    void HardMemory::dispose() {
        m_allocator(0, m_data);
    }

    void HardMemory::expect(size_t size) {
        if (size > m_capacity) {
            m_allocator(size, m_data);
            m_capacity = size;
        }
    }

    void HardMemory::shrink(size_t size) {
        if (size < m_capacity) {
            m_allocator(size, m_data);
            m_capacity = size;
        }
    }

    void HardMemory::resize(size_t size) {
        if (size != m_capacity) {
            m_allocator(size, m_data);
            m_capacity = size;
        }
    }

    void HardMemory::swap(HardMemory::self &other) {
        std::swap(self::m_device, other.m_device);
        std::swap(self::m_capacity, other.m_capacity);
        std::swap(self::m_data, other.m_data);
        std::swap(self::m_allocator, other.m_allocator);
    }

    HardMemory::HardMemory(HardMemory::self &&other) noexcept {
        self::swap(other);
    }

    HardMemory &HardMemory::operator=(HardMemory::self &&other) noexcept {
        self::swap(other);
        return *this;
    }
}
