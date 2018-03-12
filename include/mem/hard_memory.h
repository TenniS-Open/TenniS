//
// Created by lby on 2018/2/11.
//

#ifndef TENSORSTACK_MEM_HARD_MEMORY_H
#define TENSORSTACK_MEM_HARD_MEMORY_H

#include <cstddef>

#include "global/device.h"
#include "global/allocator.h"

namespace ts {
    /**
     * Hardware memory
     */
    class HardMemory {
    public:
        using self = HardMemory;    ///< self class

        HardMemory(const self &) = delete;

        const HardMemory &operator=(const self &) = delete;

        /**
         * Initialize hardware memory
         * @param device running @sa Device
         */
        explicit HardMemory(const Device &device);

        /**
         * Initialize hardware memory
         * @param device running @sa Device
         * @param size expected size
         */
        explicit HardMemory(const Device &device, size_t size);

        /**
         * Initialize hardware memory
         * @param device running @sa Device
         * @param allocator memory allocator @see HardAllocator
         */
        explicit HardMemory(const Device &device, const HardAllocator &allocator);

        /**
         * Initialize hardware memory
         * @param device running @sa Device
         * @param allocator memory allocator @see HardAllocator
         * @param size expected size
         */
        explicit HardMemory(const Device &device, const HardAllocator &allocator, size_t size);

        ~HardMemory();

        /**
         * Moving constructed function
         * @param other other memory
         */
        HardMemory(self &&other) noexcept;

        /**
         * Moving assignment function
         * @param other other memory
         */
        HardMemory &operator=(self &&other) noexcept;

        /**
         * Swap to other object
         * @param other
         */
        void swap(self &other);

        /**
         * Dispose all hardware memory
         */
        void dispose();

        /**
         * expend memory size to param size
         * @param size expected size
         */
        void expect(size_t size);

        /**
         * shrink memory size to param size
         * @param size expected size
         */
        void shrink(size_t size);

        /**
         * shrink resize size to param size
         * @param size expected size
         */
        void resize(size_t size);

        /**
         * Runing deivce
         * @return running @sa Device
         */
        const Device &device() const { return m_device; }

        /**
         * Get memory capacity on hardware
         * @return return memory capacity
         */
        size_t capacity() const { return m_capacity; }

        /**
         * Get memory start pointer
         * @return memory start pointer
         */
        void *data() { return m_data; }

        /**
         * Get memory start pointer
         * @return memory start pointer
         */
        const void *data() const { return m_data; }

        /**
         * Get memory start pointer
         * @return memory start pointer
         */
        template<typename T>
        T *data() { return reinterpret_cast<T *>(self::data()); }

        /**
         * Get memory start pointer
         * @return memory start pointer
         */
        template<typename T>
        const T *data() const { return reinterpret_cast<const T *>(self::data()); }

    private:
        Device m_device;                         ///< running device
        size_t m_capacity = 0;                   ///< memory capacity
        void *m_data = nullptr;                ///< memory start pointer
        HardAllocator m_allocator = nullptr;    ///< memory allocator
    };

    /**
     * Swap two objects
     * @param mem1 first object
     * @param mem2 second object
     */
    inline void swap(HardMemory &mem1, HardMemory &mem2) { mem1.swap(mem2); }
}


#endif //TENSORSTACK_MEM_HARD_MEMORY_H
