//
// Created by kier on 2018/5/25.
//

#ifndef TENSORSTACK_CORE_TENSOR_H
#define TENSORSTACK_CORE_TENSOR_H

#include "memory.h"
#include "dtype.h"
#include "core/controller.h"
#include "module/serialization.h"

#include <vector>

namespace ts {
    using Shape = std::vector<int>;

    class Tensor : public Serializable {
    public:
        class Prototype {
        public:
            Prototype() {}

            Prototype(const Shape &sizes) : m_sizes(sizes) {}

            Prototype(DTYPE dtype, const Shape &sizes) : m_dtype(dtype), m_sizes(sizes) {}

            explicit Prototype(DTYPE dtype) : m_dtype(dtype) {}

            DTYPE dtype() const { return m_dtype; }

            size_t dims() const { return m_sizes.size(); }

            const Shape &sizes() const { return m_sizes; }

            int type_bytes() const { return ts::type_bytes(m_dtype); }

            int count() const { return count(m_sizes); };

            static int count(const Shape &shape) {
                int prod = 1;
                for (int _size : shape) prod *= _size;
                return prod;
            }

        private:
            DTYPE m_dtype = VOID;
            std::vector<int> m_sizes = {};  ///< ?in reversed mode?
            // std::string m_layout; ///< NCHW or NHWC
        };

        using self = Tensor;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        Tensor(MemoryController::shared controller, DTYPE dtype,
               const Shape &_shape);   // allocate memory from controller

        Tensor(const MemoryDevice &device, DTYPE dtype, const Shape &_shape);

        Tensor(DTYPE dtype, const Shape &_shape);

        Tensor(MemoryController::shared controller, const Prototype &proto);   // allocate memory from controller

        Tensor(const MemoryDevice &device, const Prototype &proto);

        explicit Tensor(const Prototype &proto);

        Tensor(const Memory &memory, const Prototype &proto);

        Tensor();

        bool empty() const;

        const Device &device() const { return m_memory.device(); }

        DTYPE dtype() const { return m_proto.dtype(); }

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

        template<typename T>
        T &data(size_t i) { return m_memory.data<T>()[i]; }

        template<typename T>
        const T &data(size_t i) const { return m_memory.data<T>()[i]; }

        Tensor clone(MemoryController::shared controller) const;

        Tensor::shared clone_shared(MemoryController::shared controller) const;

        Tensor reshape(const Shape &shape) const;

        Tensor field(size_t offset) const;

        void field(size_t offset, const self &value);

        void pack(const std::vector<self> &fields);

        std::vector<self> unpack() const;

        size_t fields_count() const;

        bool packed() const;

        size_t serialize(StreamWriter &stream) const final;

        size_t externalize(StreamReader &stream) final;

    private:
        Memory m_memory;
        Prototype m_proto;

        // add field supporting structure data
        std::shared_ptr<std::vector<self>> m_fields;
    };
}


#endif //TENSORSTACK_CORE_TENSOR_H
