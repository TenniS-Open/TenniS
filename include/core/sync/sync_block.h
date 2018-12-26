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
        using key_t = _KEY;
        using value_t = _VALUE;

        enum Usage {
            READ,
            WRITE
        };

        using sync_handler = std::function<_VALUE(const _VALUE &from_value, const _KEY &from_key, const _KEY &to_key)>;

        SyncBlock(const self &) = delete;
        self &operator=(const self &) = delete;

        SyncBlock(const sync_handler &handler, bool lock)
            : m_hanlder(handler) {
            if (lock) m_mutex = std::make_shared<ts::rwmutex>();
        }

        void set(const _KEY &key, const _VALUE &value) {
            auto _write = this->lock_write();
            m_sync_values = decltype(m_sync_values)();
            m_sync_values.insert(std::make_pair(key, value));
        }

        _VALUE get(const _KEY &key) const {
            auto _read = this->lock_read();
            auto it = m_sync_values.find(key);
            if (it == m_sync_values.end()) {
                TS_LOG_ERROR << "Can not access key=" << key << eject;
            }
            return it->second;
        }

        void clear(const _KEY &key) {
            auto _write = this->lock_write();
            m_sync_values.erase(key);
        }

        _VALUE sync(const _KEY &key, Usage usage = READ) {
            switch (usage) {
                default: return sync_write(key);
                case READ: return sync_read(key);
                case WRITE: return sync_write(key);
            }
        }

        _VALUE sync_read(const _KEY &key) {
            {
                auto _read = this->lock_read();
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

        // get the value of the given key, and remove all the other's key
        _VALUE sync_write(const _KEY &key) {
            auto _write = this->lock_write();
            auto value = this->sync_insert(key);
            m_sync_values = decltype(m_sync_values)();
            m_sync_values.insert(std::make_pair(key, value));
            return value;
        }

    private:
        _VALUE sync_insert(const _KEY &key) {
            auto it = m_sync_values.find(key);
            if (it != m_sync_values.end()) {
                return it->second;
            }
            for (auto &key_value_pair : m_sync_values) {
                _VALUE value = m_hanlder(key_value_pair.second, key_value_pair.first, key);
                m_sync_values.insert(std::make_pair(key, value));
                return value;
            }
            TS_LOG_ERROR << "Can not access key=" << key << eject;
            return m_sync_values[key];    // not reachable
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

        std::unordered_map<_KEY, _VALUE> m_sync_values;
        std::shared_ptr<ts::rwmutex> m_mutex;
        sync_handler m_hanlder;
    };
}

#endif //TENSORSTACK_SYNC_SYNC_BLOCK_H
