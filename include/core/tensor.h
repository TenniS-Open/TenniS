//
// Created by kier on 2018/5/25.
//

#ifndef TENSORSTACK_CORE_TENSOR_H
#define TENSORSTACK_CORE_TENSOR_H

#include "memory.h"
#include "dtype.h"
#include "core/sync/sync_controller.h"
#include "module/serialization.h"
#include "sync/sync_memory.h"
#include <initializer_list>

#include "core/threadsafe/smart.h"

#include <vector>
#include <cassert>

#include "tensor_iterator.h"

namespace ts {
    using Shape = std::vector<int32_t>;

    inline std::string to_string(const Shape &shape) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) oss << ", ";
            oss << shape[i];
        }
        oss << "]";
        return oss.str();
    }

    using TensorMemory = SyncMemory;

    class TS_DEBUG_API TensorPrototype;

    class TS_DEBUG_API Tensor : public Serializable {
    public:
        class Prototype {
        public:
            using self = Prototype;

            Prototype() {}

            Prototype(const Shape &sizes) : m_sizes(sizes) {}

            Prototype(Shape &&sizes) : m_sizes(sizes) {}

            Prototype(DTYPE dtype, const Shape &sizes) : m_dtype(dtype), m_sizes(sizes) {}

            Prototype(DTYPE dtype, Shape &&sizes) : m_dtype(dtype), m_sizes(sizes) {}

            explicit Prototype(DTYPE dtype) : m_dtype(dtype) {}

            DTYPE dtype() const { return m_dtype; }

            size_t dims() const { return m_sizes.size(); }

            const Shape &sizes() const { return m_sizes; }

            int size(size_t i) const { return m_sizes[i]; }

            int type_bytes() const { return ts::type_bytes(m_dtype); }

            int count() const { return count(m_sizes); };

            static int count(const Shape &shape) {
                int prod = 1;
                for (int _size : shape) prod *= _size;
                return prod;
            }

        protected:
            DTYPE m_dtype = VOID;
            std::vector<int> m_sizes = {};  ///< ?in reversed mode?
            // std::string m_layout; ///< NCHW or NHWC

        public:
            Prototype(const self &other) = default;

            Prototype &operator=(const self &other) = default;

            Prototype(self &&other) {
                *this = std::move(other);
            }

            Prototype &operator=(self &&other) TS_NOEXCEPT {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
                MOVE_MEMBER(m_dtype);
                MOVE_MEMBER(m_sizes);
#undef MOVE_MEMBER
                return *this;
            }
        };

        using self = Tensor;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        Tensor(MemoryController::shared controller, DTYPE dtype,
               const Shape &_shape);   // allocate memory from controller

        Tensor(SyncMemoryController::shared controller, DTYPE dtype,
               const Shape &_shape);   // allocate memory from controller

        Tensor(SyncMemoryController::shared controller, DTYPE dtype,
               const Shape &_shape, const MemoryDevice &device);   // allocate memory from controller

        Tensor(const MemoryDevice &device, DTYPE dtype, const Shape &_shape);

        Tensor(DTYPE dtype, const Shape &_shape);

        Tensor(MemoryController::shared controller, const Prototype &proto);   // allocate memory from controller

        Tensor(SyncMemoryController::shared controller, const Prototype &proto);   // allocate memory from controller

        Tensor(SyncMemoryController::shared controller, const Prototype &proto,
               const MemoryDevice &device);   // allocate memory from controller

        Tensor(const MemoryDevice &device, const Prototype &proto);

        explicit Tensor(const Prototype &proto);

        Tensor(const Memory &memory, const Prototype &proto);

        Tensor(const SyncMemory &memory, const Prototype &proto);

        Tensor(const Smart<TensorMemory> &memory, const Prototype &proto);

        Tensor();

        Tensor(const self &) = default;

        self &operator=(const self &) = default;

        Tensor(self &&other) TS_NOEXCEPT;

        self &operator=(self &&other) TS_NOEXCEPT;

        bool empty() const;

        const MemoryDevice &device() const { return m_memory->device(); }

        DTYPE dtype() const { return m_proto.dtype(); }

        size_t dims() const { return m_proto.dims(); }

        const Shape &sizes() const { return m_proto.sizes(); }

        int size(size_t i) const { return m_proto.size(i); }

        int count() const { return m_proto.count(); };

        const Prototype &proto() const { return m_proto; }

        void *data() { return m_memory->data(); }

        const void *data() const { return m_memory->data(); }

        template<typename T>
        T *data() { return m_memory->data<T>(); }

        template<typename T>
        const T *data() const { return m_memory->data<T>(); }

        template<typename T>
        T &data(size_t i) { return m_memory->data<T>()[i]; }

        template<typename T>
        const T &data(size_t i) const { return m_memory->data<T>()[i]; }

        template<typename T>
        T &data(int i) { return m_memory->data<T>()[i]; }

        template<typename T>
        const T &data(int i) const { return m_memory->data<T>()[i]; }

        Tensor clone() const;

        Tensor clone(MemoryController::shared controller) const;

        Tensor clone(SyncMemoryController::shared controller) const;

        Tensor clone(SyncMemoryController::shared controller, const MemoryDevice &device) const;

        Tensor::shared clone_shared() const;

        Tensor::shared clone_shared(MemoryController::shared controller) const;

        Tensor::shared clone_shared(SyncMemoryController::shared controller) const;

        Tensor::shared clone_shared(SyncMemoryController::shared controller, const MemoryDevice &device) const;

        Tensor reshape(const Shape &shape) const;

        Tensor field(size_t offset) const;

        void field(size_t offset, const self &value);

        void pack(const std::vector<self> &fields);

        std::vector<self> unpack() const;

        size_t fields_count() const;

        bool packed() const;

        void refield(size_t size);

        size_t serialize(StreamWriter &stream) const final;

        /**
         *
         * @param stream
         * @return
         * @context: MemoryController, to give where new memory alloc
         */
        size_t externalize(StreamReader &stream) final;

        HypeShape hype_shape() const { return HypeShape(this->sizes()); }

        TensorMemory::shared locked() { return m_memory->locked(); }

        /**
         * @return weak memory
         */
        Memory sync() { return m_memory->sync(); }

        /**
         * @return weak memory
         */
        Memory sync() const { return m_memory->sync(); }

        /**
         * @return weak memory
         */
        Memory sync(const MemoryDevice &device) { return m_memory->sync(device); }

        /**
         * @return weak tensor, can not used in long time
         */
        Tensor view(const MemoryDevice &device) const;

        /**
         * weak tensor can not used in long time
         * @return weak tensor
         */
        Tensor weak() const;

        bool has_shape(const Shape &shape) const;

        bool has_shape(const std::initializer_list<int> &shape) const {
            return has_shape(Shape(shape.begin(), shape.end()));
        }

        Tensor flatten(int dim = 0) const;

        bool has_empty_shape() const;

        bool has_shape(int arg0) const;

        bool has_shape(int arg0, int arg1) const;

        bool has_shape(int arg0, int arg1, int arg2) const;

        bool has_shape(int arg0, int arg1, int arg2, int arg3) const;

        bool has_shape(int arg0, int arg1, int arg2, int arg3, int arg4) const;

        bool has_shape(int arg0, int arg1, int arg2, int arg3, int arg4, int arg5) const;

        bool has_shape(int arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6) const;

        bool has_shape(int arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7) const;

        bool has_shape(int arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8) const;

        bool has_shape(int arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8,
                       int arg9) const;

    private:
        Smart<TensorMemory> m_memory;
        Prototype m_proto;

        // add field supporting structure data
        std::vector<self> m_fields;
    };

    class TS_DEBUG_API TensorPrototype : public Tensor::Prototype {
    public:
        using self = TensorPrototype;
        using supper = Tensor::Prototype;

        TensorPrototype(const Tensor &tensor);

        TensorPrototype() : supper() {}

        TensorPrototype(const Shape &sizes) : supper(sizes) {}

        TensorPrototype(Shape &&sizes) : supper(std::move(sizes)) {}

        TensorPrototype(DTYPE dtype, const Shape &sizes) : supper(dtype, sizes) {}

        TensorPrototype(DTYPE dtype, Shape &&sizes) : supper(dtype, std::move(sizes)) {}

        explicit TensorPrototype(DTYPE dtype) : supper(dtype) {}

        TensorPrototype(const supper &other) : supper(other) {}

        TensorPrototype(supper &&other) : supper(std::move(other)) {}

        TensorPrototype &operator=(const supper &other) { supper::operator=(other); return *this; };

        TensorPrototype &operator=(supper &&other) { supper::operator=(std::move(other)); return *this; };

        self field(size_t offset) const;

        void field(size_t offset, const supper &value);

        void pack(const std::vector<supper> &fields);

        std::vector<supper> unpack() const;

        size_t fields_count() const;

        bool packed() const;

        void refield(size_t size);

    private:
        std::vector<supper> m_fields;

    public:
        TensorPrototype(const self &other) = default;

        TensorPrototype &operator=(const self &other) = default;

        TensorPrototype(self &&other) {
            *this = std::move(other);
        }

        TensorPrototype &operator=(self &&other) TS_NOEXCEPT {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
            MOVE_MEMBER(m_dtype);
            MOVE_MEMBER(m_sizes);
            MOVE_MEMBER(m_fields);
#undef MOVE_MEMBER
            return *this;
        }
    };
}


#endif //TENSORSTACK_CORE_TENSOR_H
