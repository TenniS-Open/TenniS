//
// Created by lby on 2018/2/11.
//

#include "mem/hard_memory.h"

#include <cstdlib>
#include <utility>
#include <cassert>

namespace ts {
    HardMemory::HardMemory(const Device &device)
            : m_device(device) {
        m_allocator = QueryAllocator(device);
        assert(m_allocator != nullptr);
    }

    HardMemory::HardMemory(const HardAllocator &allocator) {
        m_allocator = allocator;
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

    void HardMemory::swap(self &other) {
        std::swap(self::m_device, other.m_device);
        std::swap(self::m_capacity, other.m_capacity);
        std::swap(self::m_data, other.m_data);
        std::swap(self::m_allocator, other.m_allocator);
    }

    HardMemory::HardMemory(self &&other) noexcept {
        self::swap(other);
    }

    HardMemory &HardMemory::operator=(self &&other) noexcept {
        self::swap(other);
        return *this;
    }

    void swap(HardMemory &mem1, HardMemory &mem2) {
        mem1.swap(mem2);
    }
}
