//
// Created by lby on 2018/2/11.
//

#include "core/hard_memory.h"

#include <cstdlib>
#include <utility>


#include "utils/assert.h"

namespace ts {
    HardMemory::HardMemory(const MemoryDevice &device)
            : m_device(device) {
        m_allocator = HardAllocator::Query(device.type());
        TS_AUTO_CHECK(m_allocator != nullptr);
    }

    HardMemory::HardMemory(const MemoryDevice &device, size_t size)
            : HardMemory(device) {
        this->resize(size);
    }

    HardMemory::HardMemory(const MemoryDevice &device, const HardAllocator::function &allocator)
        : m_device(device), m_allocator(allocator){
        TS_AUTO_CHECK(m_allocator != nullptr);
    }

    HardMemory::HardMemory(const MemoryDevice &device, const HardAllocator::function &allocator, size_t size)
            : HardMemory(device, allocator) {
        this->resize(size);
    }

    HardMemory::~HardMemory() {
        if (m_allocator) m_allocator(m_device.id(), 0, m_data, 0);
    }

    void HardMemory::dispose() {
        m_allocator(m_device.id(), 0, m_data, 0);
        m_data = nullptr;
    }

    void HardMemory::expect(size_t size) {
        if (size > m_capacity) {
            m_data = m_allocator(m_device.id(), size, m_data, m_capacity);
            m_capacity = size;
        }
    }

    void HardMemory::shrink(size_t size) {
        if (size < m_capacity) {
            m_data = m_allocator(m_device.id(), size, m_data, m_capacity);
            m_capacity = size;
        }
    }

    void HardMemory::resize(size_t size) {
        if (size != m_capacity) {
            m_data = m_allocator(m_device.id(), size, m_data, 0);
            m_capacity = size;
        }
    }

    void HardMemory::swap(self &other) {
        std::swap(this->m_device, other.m_device);
        std::swap(this->m_capacity, other.m_capacity);
        std::swap(this->m_data, other.m_data);
        std::swap(this->m_allocator, other.m_allocator);
    }

	HardMemory::HardMemory(self &&other) TS_NOEXCEPT{
        this->swap(other);
    }

    HardMemory &HardMemory::operator=(self &&other) TS_NOEXCEPT {
        this->swap(other);
        return *this;
    }
}
