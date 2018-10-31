//
// Created by seeta on 2018/5/25.
//

#ifndef TENSORSTACK_CORE_TENSOR_H
#define TENSORSTACK_CORE_TENSOR_H

#include "memory.h"
#include "type.h"
#include "core/controller.h"

#include <vector>

namespace ts {
    using Shape = std::vector<int>;

    class Tensor {
    public:
        class Prototype {
        public:
            Prototype() {}

            Prototype(const Shape &sizes) : m_sizes(sizes) {}

            Prototype(TYPE type, const Shape &sizes) : m_type(type), m_sizes(sizes) {}

            explicit Prototype(TYPE type) : m_type(type) {}

            TYPE type() const { return m_type; }

            size_t dims() const { return m_sizes.size(); }

            const Shape &sizes() const { return m_sizes; }

            int type_bytes() const { return ts::type_bytes(m_type); }

            int count() const { return count(m_sizes); };

            static int count(const Shape &shape) {
                int prod = 1;
                for (int _size : shape) prod *= _size;
                return prod;
            }

        private:
            TYPE m_type = VOID;
            std::vector<int> m_sizes = {};  ///< ?in reversed mode?
            // std::string m_layout; ///< NCHW or NHWC
        };

        using self = Tensor;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        Tensor(MemoryController::shared controller, TYPE type,
               const Shape &_shape);   // allocate memory from controller

        Tensor(const MemoryDevice &device, TYPE type, const Shape &_shape);

        Tensor(TYPE type, const Shape &_shape);

        Tensor(MemoryController::shared controller, const Prototype &proto);   // allocate memory from controller

        Tensor(const MemoryDevice &device, const Prototype &proto);

        explicit Tensor(const Prototype &proto);

        Tensor();

        bool empty() const;

        const Device &device() const { return m_memory.device(); }

        TYPE type() const { return m_proto.type(); }

        size_t dims() const { return m_proto.dims(); }

        const Shape &sizes() const { return m_proto.sizes(); }

        int count() const { return m_proto.count(); };

        const Prototype &proto() const { return m_proto; }

        void *data() { return m_memory.data(); }

        const void *data() const { return m_memory.data(); }

        template<typename T>
        T *data() { return m_memory.data<T>(); }

        template<typename T>
        const T *data() const { return m_memory.data<T>(); }

        Tensor clone(MemoryController::shared controller) const;

        Tensor::shared clone_shared(MemoryController::shared controller) const;

        Tensor reshape(const Shape &shape) const;

    private:
        Memory m_memory;
        Prototype m_proto;
    };
}


#endif //TENSORSTACK_CORE_TENSOR_H
