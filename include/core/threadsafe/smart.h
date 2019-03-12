//
// Created by kier on 2019/2/14.
//

#ifndef TENSORSTACK_CORE_THREADSAFE_SMART_H
#define TENSORSTACK_CORE_THREADSAFE_SMART_H

#include "utils/except.h"
#include "core/memory.h"

#include <atomic>

namespace ts {

    template <typename T>
    class Counter {
    public:
        using self = Counter;
        using Object = T;

        Counter() = default;

        Counter(Object *object, int count)
            : object(object), use_count(count) {}

        Counter(const Object &object) : self(new Object(object), 1) {}

        ~Counter() {
            if (object) delete object;
        }

        Counter(const self &) = delete;

        Counter &operator=(const self &) = delete;

        Counter(self &&other) {
            *this == std::move(other);
        }

        Counter &operator==(self &&other) TS_NOEXCEPT {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
            MOVE_MEMBER(object);
            MOVE_MEMBER(use_count);
#undef MOVE_MEMBER
            return *this;
        }

        Object *object = nullptr;
        int use_count = 0;
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

        Smart() : self(Object()) {}

        Smart(const Object &object)
                : m_mode(SMART), m_counted(new CountedObject(object)) {}

        Smart(Object *object, SmartMode mode = SMART)
                : m_mode(mode), m_counted(new CountedObject(object, mode == SMART ? 1 : 0)) {}

        Smart(const self &other) {
            *this = other;
        }

        Smart &operator=(const self &other) {
            if (this == &other) return *this;
            this->dispose();
            this->m_mode = other.m_mode;
            this->m_counted = other.m_counted;
            if (m_counted && m_mode == SMART) {
                m_counted->use_count++;
            }
            return *this;
        }

        ~Smart() {
            dispose();
        }

        Smart(self &&other) {
            this->swap(other);
        }

        Smart &operator=(self &&other) TS_NOEXCEPT {
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

        Object &operator*() {
            if (m_counted == nullptr) throw NullPointerException();
            return *m_counted->object;
        }

        const Object &operator*() const {
            if (m_counted == nullptr) throw NullPointerException();
            return *m_counted->object;
        }

        Object *operator->() {
            if (m_counted == nullptr) throw NullPointerException();
            return m_counted->object;
        }

        const Object *operator->() const {
            if (m_counted == nullptr) throw NullPointerException();
            return m_counted->object;
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

    template <typename T, typename ...Args>
    Smart<T> make_smart(Args &&...args) {
        return Smart<T>(new T(std::forward<Args>(args)...), SMART);
    }
}

#endif //TENSORSTACK_CORE_THREADSAFE_SMART_H
