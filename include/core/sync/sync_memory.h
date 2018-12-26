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
        using Block = SyncBlock<MemoryDevice, Memory>;

        enum Usage {
            READ,
            WRITE,
        };

        SyncMemory(Block::sync_handler handler, bool lock, const MemoryDevice &device, const Memory &memory) {
            m_sync_memory = std::make_shared<Block>(handler, lock);
            m_sync_memory->set(device, memory);
        }

        void set(const MemoryDevice &device, const Memory &memory) {
            m_sync_memory->set(device, memory);
        }

        // write mode may cause data missmatch in multi thread access, use set instead write
        // write mode is only work for in workbench memory controll
        Memory sync(const MemoryDevice &device, Usage usage = READ) {
            switch (usage) {
                default: return m_sync_memory->sync(device, Block::READ);
                case READ: return m_sync_memory->sync(device, Block::READ);
                case WRITE: return m_sync_memory->sync(device, Block::WRITE);
            }
        }
    private:
        std::shared_ptr<Block> m_sync_memory;
    };
}


#endif //TENSORSTACK_SYNC_MEMORY_H
