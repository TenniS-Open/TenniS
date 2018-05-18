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
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        Memory(const std::shared_ptr<HardMemory> &hard);

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        Memory(std::shared_ptr<HardMemory> &&hard);

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

        /**
         * Moving constructed function
         * @param other other object
         */
        Memory(const self &other) TS_NOEXCEPT = default;

        /**
         * Moving assignment function
         * @param other other object
         */
        Memory &operator=(const self &other) TS_NOEXCEPT = default;

        /**
         * Moving constructed function
         * @param other other object
         */
        Memory(self &&other) TS_NOEXCEPT;

        /**
         * Moving assignment function
         * @param other other object
         */
        Memory &operator=(self &&other) TS_NOEXCEPT;

        /**
         * Swap to other object
         * @param other
         */
        void swap(self &other);

        /**
         * Get size of memory
         * @return size of memory
         */
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
        T *data() { return reinterpret_cast<T *>(this->data()); }

        /**
         * Get memory pointer
         * @return memory pointer
         */
        template<typename T>
        const T *data() const { return reinterpret_cast<const T *>(this->data()); }

        /**
         * Set callback when memory will be free
         * @param dtor destructor
         * @param data param will pass to destructor
         * @note the use_count will reset after this API
         * @note use one of use_count or this API to control memory, do not use them both
         */
        void destructor(const std::function<void(void*)> &dtor, void *data);

        /**
         * Set callback when memory will be free
         * @param dtor destructor
         * @param data param will pass to destructor
         * @note the use_count will reset after this API
         * @note use one of use_count or this API to control memory, do not use them both
         */
        void destructor(const std::function<void(void)> &dtor);

        /**
         * return use count of this memory block
         * @return use count
         */
        long use_count() const;

        /**
         * return Device of this memory
         * @return @see Device
         */
        const Device &device() const;

    private:
        std::shared_ptr<HardMemory> m_hard = nullptr;  ///< hardware memory
        size_t m_size = 0;                              ///< sizeof this memory block
        size_t m_shift = 0;                             ///< shift from start pointer
        std::shared_ptr<void> m_usage = nullptr;      ///< for memory usage count
    };

    /**
     * Swap two objects
     * @param obj1 first object
     * @param obj2 second object
     */
    inline void swap(Memory &obj1, Memory &obj2) {obj1.swap(obj2);}

    void memcpy(Memory &dst, const Memory &src, size_t size);

}


#endif //TENSORSTACK_MEM_MEMORY_H
