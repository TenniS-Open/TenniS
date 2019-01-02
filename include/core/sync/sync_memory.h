//
// Created by kier on 2018/12/26.
//

#ifndef TENSORSTACK_SYNC_SYNC_MEMORY_H
#define TENSORSTACK_SYNC_SYNC_MEMORY_H

#include <core/memory.h>
#include "sync_block.h"

namespace ts {
    class SyncMemory {
    public:
        using self = SyncMemory;
        using shared = std::shared_ptr<self>;

        using Block = SyncBlock<MemoryDevice, Memory>;

        static Block::value_t dynamic_sync_handler(const Block::value_t &from_memory,
                                                   const Block::key_t &from_device,
                                                   const Block::key_t &to_device) {
            Memory to_memory(to_device, from_memory.size());
            memcpy(to_memory, from_memory);
            return to_memory;
        }

        SyncMemory(const Memory &memory, bool lock, Block::sync_handler handler) {
            m_sync_memory = std::make_shared<Block>(memory.device(), memory, handler, lock);
        }

        SyncMemory(const Memory &memory, bool lock = false)
            : SyncMemory(memory, lock, dynamic_sync_handler){}

        SyncMemory(const MemoryDevice &device, size_t size, bool lock = false)
            : SyncMemory(Memory(device, size), lock) {}

        SyncMemory(size_t size, bool lock = false)
                : SyncMemory(Memory(size), lock) {}

        SyncMemory(bool lock = false)
                : SyncMemory(0, lock) {}

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        SyncMemory(const HardMemory::shared &hard, size_t size, size_t shift = 0)
                : SyncMemory(Memory(hard, size, shift)) {}

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        SyncMemory(HardMemory::shared &&hard, size_t size, size_t shift = 0)
                : SyncMemory(Memory(std::move(hard), size, shift)) {}

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        SyncMemory(const HardMemory::shared &hard)
                : SyncMemory(Memory(hard)) {}

        /**
         * Initialize Memory
         * @param hard ready memory
         * @param size sizeof the memory block
         * @param shift shift from start pointer
         */
        SyncMemory(HardMemory::shared &&hard)
                : SyncMemory(Memory(std::move(hard))) {}

        void set(const MemoryDevice &device, const Memory &memory) {
            m_sync_memory->set(device, memory);
        }

        /**
         * Moving constructed function
         * @param other other object
         */
        SyncMemory(const self &other) TS_NOEXCEPT = default;

        /**
         * Moving assignment function
         * @param other other object
         */
        SyncMemory &operator=(const self &other) TS_NOEXCEPT = default;

        /**
         * Moving constructed function
         * @param other other object
         */
        SyncMemory(self &&other) TS_NOEXCEPT {
            this->swap(other);
        }

        /**
         * Moving assignment function
         * @param other other object
         */
        SyncMemory &operator=(self &&other) TS_NOEXCEPT {
            this->swap(other);
            return *this;
        }

        /**
         * Swap to other object
         * @param other
         */
        void swap(self &other) {
            std::swap(this->m_sync_memory, other.m_sync_memory);
        }

        /**
         * Get size of memory
         * @return size of memory
         */
        size_t size() const { return m_sync_memory->value().size(); }

        /**
         * Get memory pointer
         * @return memory pointer
         */
        void *data() {
            auto default_value = m_sync_memory->value();
            return default_value.data();
        }

        /**
         * Get memory pointer
         * @return memory pointer
         */
        const void *data() const { return m_sync_memory->value().data(); }

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
         * return Device of this memory
         * @return @see Device
         */
        const MemoryDevice &device() const { return  m_sync_memory->key(); }

        // following sync part
        Memory sync() const {
            return m_sync_memory->value();
        }

        Memory sync(const MemoryDevice &device) {
            return m_sync_memory->sync(device);
        }

        void broadcast(const MemoryDevice &device) {
            m_sync_memory->broadcast(device);
        }

        shared locked() {
            std::shared_ptr<self> locked_self(new self(m_sync_memory->locked()));
            return locked_self;
        }

    private:
        SyncMemory(std::shared_ptr<Block> sync_memory) : m_sync_memory(std::move(sync_memory)) {}

        std::shared_ptr<Block> m_sync_memory;
    };

    /**
     * copy memory in device or cross devices
     * @param dst the dst memory
     * @param src the src memory
     * @param size copy size
     */
    inline void memcpy(SyncMemory &dst, const SyncMemory &src, size_t size) {
        auto locked_dst = dst.locked();
        auto locked_value = locked_dst->sync();
        memcpy(locked_value, src.sync(), size);
    }

    /**
     * copy memory in device or cross devices, copy src size
     * @param dst the dst memory
     * @param src the src memory
     */
    inline void memcpy(SyncMemory &dst, const SyncMemory &src)  {
        auto locked_dst = dst.locked();
        auto locked_value = locked_dst->sync();
        memcpy(locked_value, src.sync());
    }
}


#endif //TENSORSTACK_SYNC_MEMORY_H
