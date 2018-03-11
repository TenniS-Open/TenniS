//
// Created by lby on 2018/3/11.
//

#ifndef TENSORSTACK_MEM_MEMORY_H
#define TENSORSTACK_MEM_MEMORY_H

#include "global/device.h"
#include "hard_memory.h"

#include <memory>

namespace ts {

    /**
     * Memory, directly memory on specific device
     */
    class Memory {
    public:
        using self = Memory;    ///< self class

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        Memory(const std::shared_ptr<HardMemory> &hard, size_t size, size_t shift = 0);

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        Memory(std::shared_ptr<HardMemory> &&hard, size_t size, size_t shift = 0);

        /**
         * Initialize Memory
         * @param device running device
         * @param size sizeof this memory block
         */
        Memory(const Device &device, size_t size);

        /**
         * Initialize Memory, with cpu zero memory
         * @param size sizeof the memory block
         */
        explicit Memory(size_t size);

        size_t size() const { return m_size; }

        /**
         * Get memory pointer
         * @return memory pointer
         */
        void *data() { return m_hard->data<char>() + m_shift; }

        /**
         * Get memory pointer
         * @return memory pointer
         */
        const void *data() const { return m_hard->data<char>() + m_shift; }

        /**
         * Get memory pointer
         * @return memory pointer
         */
        template<typename T>
        T *data() { return reinterpret_cast<T *>(self::data()); }

        /**
         * Get memory pointer
         * @return memory pointer
         */
        template<typename T>
        const T *data() const { return reinterpret_cast<const T *>(self::data()); }

    private:
        std::shared_ptr<HardMemory> m_hard; ///< hardware memory
        size_t m_size;  ///< sizeof this memory block
        size_t m_shift; ///< shift from start pointer
    };

    void memcpy(const Memory &dst, const Memory &src, size_t size);

    void memset(Memory &mem, int val, size_t size);

}


#endif //TENSORSTACK_MEM_MEMORY_H
