//
// Created by seeta on 2018/5/25.
//

#ifndef TENSORSTACK_TENSOR_TENSOR_H
#define TENSORSTACK_TENSOR_TENSOR_H

#include "mem/memory.h"
#include "type.h"
#include <vector>
#include <mem/controller.h>

namespace ts {
    class tensor {
    public:
        class prototype {
        public:
            prototype() {}

            prototype(const std::vector<int> &sizes) : m_sizes(sizes) {}

            prototype(TYPE type, const std::vector<int> &sizes) : m_type(type), m_sizes(sizes) {}

            explicit prototype(TYPE type) : m_type(type) {}

            TYPE type() const { return m_type; }

            size_t dims() const { return m_sizes.size(); }

            const std::vector<int> &sizes() const { return m_sizes; }

            int type_bytes() const { return ts::type_bytes(m_type); }

            int count() const { return count(m_sizes); };

            static int count(const std::vector<int> &_shape) {
                int prod = 1;
                for (int _size : _shape) prod *= _size;
                return prod;
            }

        private:
            TYPE m_type = VOID;
            std::vector<int> m_sizes = {};
            // std::string m_layout; ///< NCHW or NHWC
        };

        tensor(MemoryController::shared &controller, TYPE type,
               const std::vector<int> &_shape);   // allocate memory from controller

        tensor(const Device &device, TYPE type, const std::vector<int> &_shape);

        tensor(TYPE type, const std::vector<int> &_shape);

        tensor(MemoryController::shared &controller, const prototype &proto);   // allocate memory from controller

        tensor(const Device &device, const prototype &proto);

        explicit tensor(const prototype &proto);

        const Device &device() const { return m_memory.device(); }

        TYPE type() const { return m_proto.type(); }

        void *data() { return m_memory.data(); }

        const void *data() const { return m_memory.data(); }

        template<typename T>
        void *data() { return m_memory.data<T>(); }

        template<typename T>
        const void *data() const { return m_memory.data<T>(); }

    private:
        Memory m_memory;
        prototype m_proto;
    };
}


#endif //TENSORSTACK_TENSOR_TENSOR_H
