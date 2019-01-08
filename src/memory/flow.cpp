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
        auto hard_free = std::bind(hard_allocator, device.id(), 0, _1);
        auto pot_allocator = [hard_allocator, device, hard_free](size_t size) -> std::shared_ptr<void> {
//            auto log_allocator = [=](int i, size_t s, void *m) -> void *{
//                auto new_mem = hard_allocator(i, s, m);
//                TS_LOG_DEBUG << "CPU malloc(" << size << "): " << new_mem;
//                return new_mem;
//            };
//
//            auto log_free = [=](void *m) -> void {
//                hard_free(m);
//                TS_LOG_DEBUG << "CPU free(" << 0 << "): " << m;
//            };
            return std::shared_ptr<void>(hard_allocator(device.id(), size, nullptr), hard_free);
        };

        m_impl->m_device = device;
        m_impl->m_vat = std::make_shared<Vat>(pot_allocator);
        m_impl->m_managed_allocator = [this](int, size_t size, void *mem) -> void * {
            void *new_mem = nullptr;
            if (size == 0) {
                m_impl->m_vat->free(mem);
                return nullptr;
            } else if (mem != nullptr) {
                m_impl->m_vat->free(mem);
                new_mem = m_impl->m_vat->malloc(size);
            } else {
                new_mem = m_impl->m_vat->malloc(size);
            }
            return new_mem;
        };
    }

    Memory VatMemoryController::alloc(size_t size) {
        return Memory(std::make_shared<HardMemory>(m_impl->m_device, m_impl->m_managed_allocator, size));
    }
}