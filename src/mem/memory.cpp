//
// Created by lby on 2018/3/11.
//

#include "mem/memory.h"

namespace ts {
    Memory::Memory(const std::shared_ptr<HardMemory> &hard, size_t size, size_t shift)
            : m_hard(hard), m_size(size), m_shift(shift) {
    }

    Memory::Memory(std::shared_ptr<HardMemory> &&hard, size_t size, size_t shift)
            : m_hard(hard), m_size(size), m_shift(shift) {
    }

    Memory::Memory(const Device &device, size_t size)
            : m_hard(new HardMemory(device, size)), m_size(size), m_shift(0) {
    }

    Memory::Memory(size_t size)
            : m_hard(new HardMemory(Device(), size)), m_size(size), m_shift(0) {
    }
}