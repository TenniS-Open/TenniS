//
// Created by lby on 2018/3/11.
//

#include "mem/memory.h"

namespace ts {
    Memory::Memory(const std::shared_ptr<HardMemory> &hard, size_t size, size_t shift)
            : m_hard(hard), m_size(size), m_shift(shift) {
    }

    Memory::Memory(std::shared_ptr<HardMemory> &&hard, size_t size, size_t shift)
            : m_hard(std::move(hard)), m_size(size), m_shift(shift) {
    }

    Memory::Memory(const Device &device, size_t size)
            : m_hard(new HardMemory(device, size)), m_size(size), m_shift(0) {
    }

    Memory::Memory(size_t size)
            : m_hard(new HardMemory(Device(), size)), m_size(size), m_shift(0) {
    }

    void Memory::destructor(const std::function<void(void *)> &dtor, void *data) {
        m_usage.reset(data, dtor);
    }

    void Memory::swap(Memory::self &other) {
        std::swap(this->m_hard, other.m_hard);
        std::swap(this->m_size, other.m_size);
        std::swap(this->m_shift, other.m_shift);
        std::swap(this->m_usage, other.m_usage);
    }

    Memory::Memory(Memory::self &&other) TS_NOEXCEPT {
        this->swap(other);
    }

    Memory &Memory::operator=(Memory::self &&other) TS_NOEXCEPT {
        this->swap(other);
        return *this;
    }

    Memory::Memory(const std::shared_ptr<HardMemory> &hard)
            : Memory(hard, hard->capacity()) {
    }

    Memory::Memory(std::shared_ptr<HardMemory> &&hard)
            : Memory(std::move(hard), hard->capacity()) {
    }
}