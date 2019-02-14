//
// Created by kier on 2019/2/14.
//

#ifndef TENSORSTACK_CORE_THREADSAFE_SMART_H
#define TENSORSTACK_CORE_THREADSAFE_SMART_H

#include "utils/except.h"
#include "core/memory.h"

namespace ts {

    template <typename T>
    class Counter {
    public:
        using self = Counter;
        using Object = T;

        Counter() = default;

        Counter(const Object &object, int count = 1)
                : object(object), use_count(count) {}

        Object object;
        std::atomic_int use_count;
    };

    enum SmartMode {
        SMART = 0,
        MANUALLY = 1,
    };

    // This class can be copy in object
    template <typename T>
    class Smart {
    public:
        using self = Smart;
        using Object = T;
        using CountedObject = Counter<Object>;

        Smart() = default;

        Smart(const Object &memory)
                : Smart(memory, SMART) {}

        Smart(const Object &memory, SmartMode mode)
                : m_mode(mode), m_counted(new CountedObject(memory, mode == SMART ? 1 : 0)) {}

        Smart(const self &other) {
            *this = other;
        }

        Smart &operator=(const self &other) {
            if (this == &other) return *this;
            this->release();
            this->m_mode = other.m_mode;
            this->m_counted = other.m_counted;
            if (m_counted && m_mode == SMART) {
                m_counted->use_count++;
            }
            return *this;
        }

        ~Smart() {
            release();
        }

        Smart(self &&other) {
            this->swap(other);
        }

        Smart &operator=(self &&other) TS_NOEXCEPT {
            this->swap(other);
            return *this;
        }

        void release() {
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

        Object &operator*() {
            if (m_counted == nullptr) throw NullPointerException();
            return m_counted->memory;
        }

        const Object &operator*() const {
            if (m_counted == nullptr) throw NullPointerException();
            return m_counted->memory;
        }

        Object *operator->() {
            if (m_counted == nullptr) throw NullPointerException();
            return &m_counted->memory;
        }

        const Object *operator->() const {
            if (m_counted == nullptr) throw NullPointerException();
            return &m_counted->memory;
        }

        self weak() const {
            return self(MANUALLY, m_counted);
        }

        self strong() const {
            if (m_counted == nullptr || m_counted->use_count <= 0) {
                throw NullPointerException();
            }
            if (m_mode == SMART) return *this;
            self strong_ptr(SMART, m_counted);
            m_counted->use_count++;
            return std::move(strong_ptr);
        }

        int use_count() const {
            if (m_counted) return m_counted->use_count;
            return 0;
        }

        operator bool() const {
            return m_counted != nullptr;
        }

    private:
        Smart(SmartMode mode, CountedObject *counted)
                : m_mode(mode), m_counted(counted) {}

        SmartMode m_mode = MANUALLY;
        CountedObject *m_counted = nullptr;
    };
}

#endif //TENSORSTACK_CORE_THREADSAFE_SMART_H
