//
// Created by lby on 2018/3/11.
//

#include "mem/memory.h"

namespace ts {
    Memory::Memory(const Device &device, size_t size, const std::shared_ptr<HardMemory> &hard, size_t shift)
        : m_device(device), m_size(size), m_hard(hard), m_shift(shift){
    }

    Memory::Memory(const Device &device, size_t size, std::shared_ptr<HardMemory> &&hard, size_t shift)
            : m_device(device), m_size(size), m_hard(hard), m_shift(shift){
    }

    Memory::Memory(const Device &device, size_t size)
            : m_device(device), m_size(size), m_hard(new HardMemory(device, size)), m_shift(0){
    }

    Memory::Memory(size_t size)
            : m_device(), m_size(size), m_shift(0){
        m_hard.reset(new HardMemory(m_device));
        m_hard->expect(size);
    }
}