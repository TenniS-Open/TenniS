//
// Created by lby on 2018/2/11.
//

#ifndef TENSORSTACK_MEM_HARD_MEMORY_H
#define TENSORSTACK_MEM_HARD_MEMORY_H

#include <cstddef>

#include "global/device.h"
#include "global/allocator.h"

namespace ts {
    class HardMemory {
    public:
        using self = HardMemory;

        HardMemory(const self &) = delete;

        const HardMemory &operator=(const self &) = delete;

        explicit HardMemory(const Device &device);

        explicit HardMemory(const HardAllocator &allocator);

        ~HardMemory();

        HardMemory(self &&other) noexcept;

        HardMemory &operator=(self &&other) noexcept;

        void swap(self &other);

        void dispose();

        void expect(size_t size);

        void shrink(size_t size);

        void resize(size_t size);

        const Device &device() const { return m_device; }

        size_t capacity() const { return m_capacity; }

        void *data() { return m_data; }

        const void *data() const { return m_data; }

    private:
        Device m_device;
        size_t m_capacity = 0;
        void *m_data = nullptr;
        HardAllocator m_allocator = nullptr;
    };

    void swap(HardMemory &mem1, HardMemory &mem2);
}


#endif //TENSORSTACK_MEM_HARD_MEMORY_H
