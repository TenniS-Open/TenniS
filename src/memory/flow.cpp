//
// Created by kier on 2019/1/8.
//

#include <memory/flow.h>
#include <global/hard_converter.h>

#include "memory/flow.h"

#include "utils/assert.h"
#include "orz/vat.h"

namespace ts {
    class VatMemoryController::Implement {
    public:
        using self = Implement;
        MemoryDevice m_device;
        HardAllocator::function m_managed_allocator;
        std::shared_ptr<Vat> m_vat;
    };

    VatMemoryController::VatMemoryController(const MemoryDevice &device) {
        TS_AUTO_CHECK(m_impl.get() != nullptr);
        auto hard_allocator = HardAllocator::Query(device.type());
        TS_CHECK(hard_allocator != nullptr) << "Can not found memory controller for " << device.type();
        using namespace std::placeholders;
        auto hard_free = std::bind(hard_allocator, device.id(), 0, _1, 0);
        auto pot_allocator = [hard_allocator, device, hard_free](size_t size) -> std::shared_ptr<void> {
            return std::shared_ptr<void>(hard_allocator(device.id(), size, nullptr, 0), hard_free);
        };

        m_impl->m_device = device;
        m_impl->m_vat = std::make_shared<Vat>(pot_allocator);
        m_impl->m_managed_allocator = [this](int, size_t new_size, void *mem, size_t mem_size) -> void * {
            void *new_mem = nullptr;
            if (new_size == 0) {
                m_impl->m_vat->free(mem);
                return nullptr;
            } else if (mem != nullptr) {
                if (mem_size > 0) {
                    TS_LOG_ERROR << "Reach the un-given code" << eject;
                } else {
                    m_impl->m_vat->free(mem);
                    new_mem = m_impl->m_vat->malloc(new_size);
                }
            } else {
                new_mem = m_impl->m_vat->malloc(new_size);
            }
            return new_mem;
        };
    }

    Memory VatMemoryController::alloc(size_t size) {
        return Memory(std::make_shared<HardMemory>(m_impl->m_device, m_impl->m_managed_allocator, size));
    }
}