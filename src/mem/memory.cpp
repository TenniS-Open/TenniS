//
// Created by lby on 2018/3/11.
//

#include <cassert>
#include "mem/memory.h"
#include "global/converter.h"

namespace ts {
    static void *const FakeUsagePtr = (void *) (0x19910929);

    static void default_usage_destructor(void *) {}

    static std::shared_ptr<void> default_usage() {
        return std::shared_ptr<void>(FakeUsagePtr, default_usage_destructor);
    }

    Memory::Memory(const std::shared_ptr<HardMemory> &hard, size_t size, size_t shift)
            : m_hard(hard), m_size(size), m_shift(shift), m_usage(default_usage()) {
    }

    Memory::Memory(std::shared_ptr<HardMemory> &&hard, size_t size, size_t shift)
            : m_hard(std::move(hard)), m_size(size), m_shift(shift), m_usage(default_usage()) {
    }

    Memory::Memory(const Device &device, size_t size)
            : m_hard(new HardMemory(device, size)), m_size(size), m_shift(0), m_usage(default_usage()) {
    }

    Memory::Memory(size_t size)
            : m_hard(new HardMemory(Device(), size)), m_size(size), m_shift(0), m_usage(default_usage()) {
    }

    void Memory::destructor(const std::function<void(void *)> &dtor, void *data) {
        m_usage.reset(data, dtor);
    }

    void Memory::destructor(const std::function<void(void)> &dtor) {
        m_usage.reset(FakeUsagePtr, [dtor](void *) -> void { dtor(); });
    }

    void Memory::swap(Memory::self &other) {
        std::swap(this->m_hard, other.m_hard);
        std::swap(this->m_size, other.m_size);
        std::swap(this->m_shift, other.m_shift);
        std::swap(this->m_usage, other.m_usage);
    }

    Memory::Memory(Memory::self &&other) TS_NOEXCEPT {
        this->swap(other);
    }

    Memory &Memory::operator=(Memory::self &&other) TS_NOEXCEPT {
        this->swap(other);
        return *this;
    }

    Memory::Memory(const std::shared_ptr<HardMemory> &hard)
            : Memory(hard, hard->capacity()) {
    }

    Memory::Memory(std::shared_ptr<HardMemory> &&hard)
            : Memory(std::move(hard), hard->capacity()) {
    }

    long Memory::use_count() const {
        return m_usage.use_count();
    }

    const Device &Memory::device() const {
        return this->m_hard->device();
    }

    void memcpy(Memory &dst, const Memory &src, size_t size) {
        assert(dst.size() >= size);
        assert(src.size() >= size);
        HardConverter converter = QueryConverter(dst.device().type(), src.device().type());
        assert(converter != nullptr);
        converter(dst.device().id(), dst.data(), src.device().id(), src.data(), size);
    }
}