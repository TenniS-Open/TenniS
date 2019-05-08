//
// Created by kier on 2019/3/16.
//

#ifndef TENSORSTACK_API_CPP_TENSOR_H
#define TENSORSTACK_API_CPP_TENSOR_H

#include "../tensor.h"

#include "except.h"
#include "dtype.h"

#include <memory>
#include <vector>
#include <numeric>

namespace ts {
    namespace api {

        using Shape = std::vector<int32_t>;

        class Tensor {
        public:
            class InFlow {
            public:
                using self = InFlow;

                InFlow() : self(TS_HOST) {}

                InFlow(ts_InFlow dtype) : raw(dtype) {}

                operator ts_InFlow() const { return raw; }

                ts_InFlow raw;
            };

            static const InFlow HOST;
            static const InFlow DEVICE;

            using self = Tensor;
            using raw = ts_Tensor;

            using shared = std::shared_ptr<self>;
            using shared_raw = std::shared_ptr<raw>;

            static self NewRef(raw *ptr) { return self(ptr); }

            Tensor(const self &) = default;

            Tensor &operator=(const self &) = default;

            raw *get_raw() const { return m_impl.get(); }

            bool operator==(std::nullptr_t) const { return get_raw() == nullptr; }

            bool operator!=(std::nullptr_t) const { return get_raw() != nullptr; }

            Tensor(std::nullptr_t) {}

            Tensor() : self(TS_VOID, {}, nullptr) {}

            Tensor(DTYPE dtype, const Shape &shape, const void *data = nullptr)
                    : self(ts_new_Tensor(shape.data(), int32_t(shape.size()), ts_DTYPE(dtype), data)) {
                TS_API_AUTO_CHECK(m_impl != nullptr);
            }

            Tensor(InFlow in_flow, DTYPE dtype, const Shape &shape, const void *data = nullptr)
                    : self(ts_new_Tensor_in_flow(in_flow, shape.data(), int32_t(shape.size()), ts_DTYPE(dtype), data)) {
                TS_API_AUTO_CHECK(m_impl != nullptr);
            }

            Shape sizes() const {
                auto shape = ts_Tensor_shape(m_impl.get());
                auto shape_len = ts_Tensor_shape_size(m_impl.get());
                return Shape(shape, shape + shape_len);
            }

            int size(int i) const {
                if (i < 0 || i >= ts_Tensor_shape_size(m_impl.get())) throw Exception("index out of range");
                return ts_Tensor_shape(m_impl.get())[i];
            }

            int size(size_t i) const {
                if (i >= size_t(ts_Tensor_shape_size(m_impl.get()))) throw Exception("index out of range");
                return ts_Tensor_shape(m_impl.get())[i];
            }

            int dims() const {
                return ts_Tensor_shape_size(m_impl.get());
            }

            DTYPE dtype() const {
                return DTYPE(ts_Tensor_dtype(m_impl.get()));
            }

            const void *data() const {
                return ts_Tensor_data(m_impl.get());
            }

            void *data() {
                return ts_Tensor_data(m_impl.get());
            }

            template<typename T>
            const T *data() const {
                return reinterpret_cast<const T *>(data());
            }

            template<typename T>
            T *data() {
                return reinterpret_cast<T *>(data());
            }

            template<typename T>
            const T &data(int i) const {
                return reinterpret_cast<const T *>(data())[i];
            }

            template<typename T>
            T &data(int i) {
                return reinterpret_cast<T *>(data())[i];
            }

            template<typename T>
            const T &data(size_t i) const {
                return reinterpret_cast<const T *>(data())[i];
            }

            template<typename T>
            T &data(size_t i) {
                return reinterpret_cast<T *>(data())[i];
            }

            self clone() const {
                auto clone_raw = ts_Tensor_clone(m_impl.get());
                TS_API_AUTO_CHECK(clone_raw != nullptr);
                return Tensor(clone_raw);
            }

            void sync_cpu() {
                TS_API_AUTO_CHECK(ts_Tensor_sync_cpu(m_impl.get()));
            }

            Tensor view(InFlow in_flow) const {
                auto casted_raw = ts_Tensor_view_in_flow(m_impl.get(), in_flow);
                TS_API_AUTO_CHECK(casted_raw != nullptr);
                return Tensor(casted_raw);
            }

            Tensor cast(DTYPE dtype) const {
                auto casted_raw = ts_Tensor_cast(m_impl.get(), ts_DTYPE(dtype));
                TS_API_AUTO_CHECK(casted_raw != nullptr);
                return Tensor(casted_raw);
            }

            int32_t count() const {
                auto shape = ts_Tensor_shape(m_impl.get());
                auto shape_len = ts_Tensor_shape_size(m_impl.get());
                return std::accumulate(shape, shape + shape_len, 1, std::multiplies<int32_t>());
            }

            Tensor reshape(const Shape &shape) const {
                auto casted_raw = ts_Tensor_reshape(m_impl.get(), shape.data(), int32_t(shape.size()));
                TS_API_AUTO_CHECK(casted_raw != nullptr);
                return Tensor(casted_raw);
            }

            Tensor field(int index) const {
                auto field_raw = ts_Tensor_field(m_impl.get(), index);
                TS_API_AUTO_CHECK(field_raw != nullptr);
                return Tensor(field_raw);
            }

            bool packed() const {
                return bool(ts_Tensor_packed(m_impl.get()));
            }

            int fields_count() const {
                return int(ts_Tensor_fields_count(m_impl.get()));
            }

            static Tensor Pack(const std::vector<Tensor> &fields) {
                std::vector<ts_Tensor*> cfields;
                for (auto &field : fields) {
                    cfields.emplace_back(field.get_raw());
                }
                return Pack(cfields.data(), int32_t(cfields.size()));
            }

            static Tensor Pack(ts_Tensor **fields, int32_t count) {
                auto packed_raw = ts_Tensor_pack(fields, count);
                TS_API_AUTO_CHECK(packed_raw != nullptr);
                return Tensor(packed_raw);
            }

            static self BorrowedRef(raw *ptr) {
                self borrowed(nullptr);
                borrowed.m_impl = shared_raw(ptr, [](raw *) {});
                return std::move(borrowed);
            }

        private:
            Tensor(raw *ptr) : m_impl(pack(ptr)) {}

            static shared_raw pack(raw *ptr) { return shared_raw(ptr, ts_free_Tensor); }

            shared_raw m_impl;
        };

        static const Tensor::InFlow HOST = TS_HOST;
        static const Tensor::InFlow DEVICE = TS_DEVICE;

        template<typename T>
        class tensor_builder {
        public:
            static Tensor build(const T &value) {
                return build(&value, 1);
            }

            static Tensor build(const std::initializer_list<T> &value) {
                return build(std::vector<T>(value.begin(), value.end()));
            }

            static Tensor build(const std::vector<T> &value) {
                return build(value.data(), value.size());
            }

            static Tensor build(const T *data, size_t count) {
                return Tensor(dtypeid<T>::id, {int(count)}, data);
            }

            static Tensor build(const std::initializer_list<T> &value, const std::vector<int> &shape) {
                return build(std::vector<T>(value.begin(), value.end()), shape);
            }

            static Tensor build(const std::vector<T> &value, const std::vector<int> &shape) {
                return build(value.data(), value.size(), shape);
            }

            static Tensor build(const T *data, size_t count, const std::vector<int> &shape) {
                auto shape_count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int32_t>());
                if (int32_t(count) != shape_count) throw Exception("Shape count mismatch.");
                return Tensor(dtypeid<T>::id, shape, data);
            }
        };

        namespace tensor {
            inline Tensor cast(DTYPE dtype, const Tensor &value) {
                return value.cast(dtype);
            }

            inline Tensor clone(DTYPE dtype, const Tensor &value) {
                Tensor dolly = value.clone();
                dolly.sync_cpu();
                return std::move(dolly);
            }

            Tensor from(const std::string &value) {
                return Tensor(TS_CHAR8, {int32_t(value.length())}, value.c_str());
            }

            template<size_t _size>
            inline Tensor from(const char (&value)[_size]) { return from(std::string(value)); }

            inline Tensor from(const char *value) { return from(std::string(value)); }

            template<typename T>
            Tensor from(const T value) { return tensor_builder<T>::build(value); }

            template<typename T>
            Tensor from(const std::initializer_list<T> &value) { return tensor_builder<T>::build(value); }

            template<typename T>
            Tensor from(const std::vector<T> &value) { return tensor_builder<T>::build(value); }

            int to_int(const Tensor &value) {
                auto cpu_value = value;
                cpu_value.sync_cpu();
                if (cpu_value.dtype() != TS_INT32) cpu_value = cast(TS_INT32, cpu_value);
                return cast(TS_INT32, cpu_value).data<int32_t>()[0];
            }

            unsigned int to_uint(const Tensor &value) {
                auto cpu_value = value;
                cpu_value.sync_cpu();
                if (cpu_value.dtype() != TS_UINT32) cpu_value = cast(TS_UINT32, cpu_value);
                return cast(TS_UINT32, cpu_value).data<uint32_t>()[0];
            }

            float to_float(const Tensor &value) {
                auto cpu_value = value;
                cpu_value.sync_cpu();
                if (cpu_value.dtype() != TS_FLOAT32) cpu_value = cast(TS_FLOAT32, cpu_value);
                return cast(TS_FLOAT32, cpu_value).data<float>()[0];
            }

            double to_double(const Tensor &value) {
                auto cpu_value = value;
                cpu_value.sync_cpu();
                if (cpu_value.dtype() != TS_FLOAT64) cpu_value = cast(TS_FLOAT64, cpu_value);
                return cast(TS_FLOAT64, cpu_value).data<double>()[0];
            }

            std::string to_string(const Tensor &value) {
                auto cpu_value = value;
                cpu_value.sync_cpu();
                if (cpu_value.dtype() != TS_CHAR8) {
                    cpu_value = cast(TS_CHAR8, cpu_value);
                }
                return std::string(cpu_value.data<char>(), size_t(cpu_value.count()));
            }

            template<typename T>
            inline Tensor build(DTYPE dtype, const T &value) {
                return cast(dtype, tensor_builder<T>::build(value));
            }

            template<typename T>
            inline Tensor build(DTYPE dtype, const std::initializer_list<T> &value) {
                return cast(dtype, tensor_builder<T>::build(value));
            }

            template<typename T>
            inline Tensor build(DTYPE dtype, const std::vector<T> &value) {
                return cast(dtype, tensor_builder<T>::build(value));
            }

            template<typename T>
            inline Tensor build(DTYPE dtype, size_t count, const T *data) {
                return cast(dtype, tensor_builder<T>::build(data, count));
            }

            template<typename T>
            inline Tensor build(DTYPE dtype, const Shape &shape, const std::initializer_list<T> &value) {
                return cast(dtype, tensor_builder<T>::build(value, shape));
            }

            template<typename T>
            inline Tensor build(DTYPE dtype, const Shape &shape, const std::vector<T> &value) {
                return cast(dtype, tensor_builder<T>::build(value, shape));
            }

            template<typename T>
            inline Tensor build(DTYPE dtype, const Shape &shape, const T *data) {
                int count = 1;
                for (auto &size : shape) count *= size;
                return cast(dtype, tensor_builder<T>::build(data, size_t(count), shape));
            }
        }
    }
}

#endif //TENSORSTACK_API_CPP_TENSOR_H
