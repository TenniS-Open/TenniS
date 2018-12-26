//
// Created by kier on 2018/12/26.
//

#ifndef TENSORSTACK_SYNC_SYNC_CONTROLLER_H
#define TENSORSTACK_SYNC_SYNC_CONTROLLER_H

#include <memory>
#include <core/device.h>
#include <core/memory.h>
#include <core/controller.h>

#include "sync_block.h"
#include "sync_memory.h"

namespace ts {
    class SyncMemoryController {
    public:
        using self = SyncMemoryController;
        using shared = std::shared_ptr<self>;  ///< smart pointer

        virtual ~SyncMemoryController() = default;
        /**
         * alloc memory with size
         * @param device device to get memory
         * @param size memory size (bytes)
         * @return allocated memory
         */
        virtual SyncMemory alloc(const MemoryDevice &device, size_t size, bool need_lock) = 0;
    };

    template <typename _MemoryController>
    class HypeSyncMemoryController
            : public SyncMemoryController,
            public std::enable_shared_from_this<HypeSyncMemoryController<_MemoryController>> {
    public:
        using self = HypeSyncMemoryController;
        using supper = SyncMemoryController;

        using shared = std::shared_ptr<self>;

        using BaseMemoryController = _MemoryController;

        static shared Make(bool need_lock) {
            return shared(new self(need_lock));
        }

    private:
        HypeSyncMemoryController(bool need_lock) : m_sync_controllers(sync_controller_handler, need_lock) {
            auto cpu_device = MemoryDevice(CPU);
            auto cpu_controller = std::make_shared<BaseMemoryController>(cpu_device);
            m_sync_controllers.set(cpu_device, cpu_controller);
        }

    public:
        void clear(const MemoryDevice &device) {
            m_sync_controllers.clear(device);
        }

        SyncMemory alloc(const MemoryDevice &device, size_t size, bool need_lock) override {
            auto controller = m_sync_controllers.sync(device);
            auto memory = controller->alloc(size);
            return SyncMemory(this->sync_handler(), need_lock, device, memory);
        }

        SyncMemory::Block::sync_handler sync_handler() {
            auto shared_this = this->shared_from_this();
            return [=](const typename SyncMemory::Block::value_t &from_memory,
                       const typename SyncMemory::Block::key_t &from_device,
                       const typename SyncMemory::Block::key_t &to_device) {
                auto controller = shared_this->m_sync_controllers.sync(to_device);
                auto to_memory = controller->alloc(from_memory.size());
                memcpy(to_memory, from_memory);
                return to_memory;
            };
        }

    private:
        using SyncControllerBlock = SyncBlock<MemoryDevice, std::shared_ptr<BaseMemoryController>>;

        SyncControllerBlock m_sync_controllers;

        static typename SyncControllerBlock::value_t sync_controller_handler(
                const typename SyncControllerBlock::value_t &,
                const typename SyncControllerBlock::key_t &,
                const typename SyncControllerBlock::key_t &device) {
            return std::make_shared<BaseMemoryController>(device);
        }
    };
}


#endif //TENSORSTACK_SYNC_SYNC_CONTROLLER_H
