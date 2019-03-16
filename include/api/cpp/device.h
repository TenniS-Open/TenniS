//
// Created by kier on 2019/3/16.
//

#ifndef TENSORSTACK_API_CPP_DEVICE_H
#define TENSORSTACK_API_CPP_DEVICE_H

#include "../device.h"

#include <string>

namespace ts {
    namespace cpp {
        class Device {
        public:
            using self = Device;
            using raw = ts_Device;

            using shared = std::shared_ptr<self>;
            using shared_raw = std::shared_ptr<raw>;

            Device(const self &) = default;

            Device &operator=(const self &) = default;

            const raw *get_raw() const {
                return &m_raw;
            }

            Device() : self("cpu", 0) {}

            Device(const std::string &type, int id = 0) {
                m_type = type;
                m_raw.id = id;
                m_raw.type = m_type.c_str();
            }

            const std::string &type() const { return m_type; }

            int id() const { return m_raw.id; }

        private:
            Device(raw *ptr) {
                m_type = ptr->type;
                m_raw.id = ptr->id;
                m_raw.type = m_type.c_str();
            }

            raw m_raw;

            std::string m_type;
        };
    }
}

#endif //TENSORSTACK_API_CPP_DEVICE_H
