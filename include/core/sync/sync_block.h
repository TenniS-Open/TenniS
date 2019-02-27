//
// Created by kier on 2018/12/26.
//

#ifndef TENSORSTACK_SYNC_SYNC_BLOCK_H
#define TENSORSTACK_SYNC_SYNC_BLOCK_H

#include <functional>
#include <unordered_map>
#include <utility>

#include <utils/mutex.h>
#include <utils/log.h>

namespace ts {
    template <typename _KEY, typename _VALUE>
    class SyncBlock {
    public:
        using self = SyncBlock;
        using shared = std::shared_ptr<self>;

        using key_t = _KEY;
        using value_t = _VALUE;

        using sync_handler = std::function<_VALUE(const _VALUE &from_value, const _KEY &from_key, const _KEY &to_key)>;

        SyncBlock(const self &) = delete;
        self &operator=(const self &) = delete;

        SyncBlock(const _KEY &key, const _VALUE &value, const sync_handler &handler, bool need_lock)
            : m_hanlder(handler) {
            if (need_lock) m_mutex = std::make_shared<ts::rwmutex>();
            m_default_key = key;
            this->set(key, value);
        }

        void set(const _KEY &key, const _VALUE &value) {
            auto _write = this->lock_write();
            if (key == m_default_key) {
                m_default_value = value;
                m_sync_values.clear();
            } else {
                m_sync_values.clear();
                m_sync_values.insert(std::make_pair(key, value));
                m_default_value = m_hanlder(value, key, m_default_key);
            }
        }

        _VALUE &get(const _KEY &key) {
            auto _read = this->lock_read();
            if (key == m_default_key) return m_default_value;
            auto it = m_sync_values.find(key);
            if (it == m_sync_values.end()) {
                TS_LOG_ERROR << "Can not access key=" << key << eject;
            }
            return it->second;
        }

        const _VALUE &get(const _KEY &key) const {
            return const_cast<self *>(this)->get(key);
        }

        void clear() {
            m_sync_values.clear();
        }

        void clear(const _KEY &key) {
            auto _write = this->lock_write();
            if (key == m_default_key) TS_LOG_ERROR << "Can not clear default key=" << key << eject;
            m_sync_values.erase(key);
        }

        _VALUE &sync(const _KEY &key) {
            {
                auto _read = this->lock_read();
                if (key == m_default_key) return m_default_value;
                auto it = m_sync_values.find(key);
                if (it != m_sync_values.end()) {
                    return it->second;
                }
            }
            {
                auto _write = this->lock_write();
                return this->sync_insert(key);
            }
        }

        void broadcast(const _KEY &key) {
            auto _write = this->lock_write();
            if (key == m_default_key) {
                m_sync_values.clear();
            } else {
                auto it = m_sync_values.find(key);
                if (it == m_sync_values.end()) {
                    TS_LOG_ERROR << "Can not broadcast key=" << key << eject;
                }
                auto value = it->second;
                m_sync_values.clear();
                m_sync_values.insert(std::make_pair(key, value));
                m_default_value = m_hanlder(value, key, m_default_key);
            }
        }

        const _KEY &key() const {
            return m_default_key;
        }

        const _VALUE &value() const {
            auto _read = this->lock_read();
            return m_default_value;
        }

        shared locked() {
            std::shared_ptr<unique_write_lock> _write(m_mutex ? new unique_write_lock(*m_mutex) : nullptr);
            std::shared_ptr<self> dolly(new self, [this](const self *ptr){
                this->m_default_value = ptr->m_default_value;
                this->m_sync_values = ptr->m_sync_values;
                delete ptr;
            });
            dolly->m_default_key = m_default_key;
            dolly->m_default_value = m_default_value;
            dolly->m_sync_values = m_sync_values;
            dolly->m_hanlder = m_hanlder;
            std::swap(dolly->m_locked, _write);
            return dolly;
        }

    private:
        SyncBlock() = default;

        _VALUE &sync_insert(const _KEY &key) {
            if (key == m_default_key) return m_default_value;
            auto it = m_sync_values.find(key);
            if (it != m_sync_values.end()) {
                return it->second;
            }
            _VALUE value = m_hanlder(m_default_value, m_default_key, key);
            auto pair_it = m_sync_values.insert(std::make_pair(key, value));
            return pair_it.first->second;
        }

        using unique_read_lock = ts::unique_read_lock<ts::rwmutex>;
        using unique_write_lock = ts::unique_write_lock<ts::rwmutex>;

        std::unique_ptr<unique_read_lock> lock_read() const {
            if (m_mutex) {
                return std::unique_ptr<unique_read_lock>(new unique_read_lock(*m_mutex));
            } else {
                return std::unique_ptr<unique_read_lock>(nullptr);
            }
        }

        std::unique_ptr<unique_write_lock> lock_write() const {
            if (m_mutex) {
                return std::unique_ptr<unique_write_lock>(new unique_write_lock(*m_mutex));
            } else {
                return std::unique_ptr<unique_write_lock>(nullptr);
            }
        }

        _KEY m_default_key;
        _VALUE m_default_value;

        std::unordered_map<_KEY, _VALUE> m_sync_values;
        std::shared_ptr<ts::rwmutex> m_mutex;
        sync_handler m_hanlder;

        std::shared_ptr<unique_write_lock> m_locked;
    };
}

#endif //TENSORSTACK_SYNC_SYNC_BLOCK_H
