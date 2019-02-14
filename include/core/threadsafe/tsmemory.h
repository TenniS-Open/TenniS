//
// Created by kier on 2019/2/14.
//

#ifndef TENSORSTACK_CORE_THREADSAFE_TSMEMORY_H
#define TENSORSTACK_CORE_THREADSAFE_TSMEMORY_H

#include "utils/except.h"
#include "core/memory.h"

namespace ts {
    using MemoryCore = Memory;

    class CountedMemory {
    public:
        CountedMemory() = default;

        CountedMemory(const MemoryCore &memory, int count = 1)
                : memory(memory), use_count(count) {}

        MemoryCore memory;
        std::atomic_int use_count;
    };

    // This class can be copy in object
    class TSMemory {
    public:
        using self = TSMemory;

        enum Mode {
            SMART = 0,
            MANUALLY = 1,
        };

        TSMemory(const MemoryCore &memory)
                : TSMemory(memory, SMART) {}

        TSMemory(const MemoryCore &memory, Mode mode)
                : m_mode(mode), m_counted(new CountedMemory(memory, 1)) {}

        TSMemory(const self &other) {
            *this = other;
        }

        TSMemory &operator=(const self &other) {
            if (this == &other) return *this;
            this->dispose();
            this->m_mode = other.m_mode;
            this->m_counted = other.m_counted;
            if (m_counted && m_mode == SMART) {
                m_counted->use_count++;
            }
            return *this;
        }

        ~TSMemory() {
            dispose();
        }

        TSMemory(self &&other) {
            this->swap(other);
        }

        TSMemory &operator=(self &&other) TS_NOEXCEPT {
            this->swap(other);
            return *this;
        }

        void dispose() {
            if (m_counted && m_mode == SMART) {
                --m_counted->use_count;
                if (m_counted->use_count <= 0) {
                    delete m_counted;
                    m_counted = nullptr;
                }
            }
        }

        void swap(self &other) {
            std::swap(m_mode, other.m_mode);
            std::swap(m_counted, other.m_counted);
        }

        MemoryCore &operator*() {
            if (m_counted == nullptr) throw NullPointerException();
            return m_counted->memory;
        }

        const MemoryCore &operator*() const {
            if (m_counted == nullptr) throw NullPointerException();
            return m_counted->memory;
        }

        MemoryCore *operator->() {
            if (m_counted == nullptr) throw NullPointerException();
            return &m_counted->memory;
        }

        const MemoryCore *operator->() const {
            if (m_counted == nullptr) throw NullPointerException();
            return &m_counted->memory;
        }

        TSMemory weak() const {
            TSMemory weak_ptr(m_mode, m_counted);
            weak_ptr.m_mode = MANUALLY;
            return weak_ptr;
        }

        int use_count() const {
            if (m_counted) return m_counted->use_count;
            return 0;
        }

    private:
        TSMemory(Mode mode, CountedMemory *counted)
                : m_mode(mode), m_counted(counted) {}

        Mode m_mode = SMART;
        CountedMemory *m_counted = nullptr;
    };
}


#endif //TENSORSTACK_TSMEMORY_H
