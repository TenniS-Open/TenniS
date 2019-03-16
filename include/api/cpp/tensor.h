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
            using self = Tensor;
            using raw = ts_Tensor;

            using shared = std::shared_ptr<self>;
            using shared_raw = std::shared_ptr<raw>;

            Tensor(const self &) = default;

            Tensor &operator=(const self &) = default;

            raw *get_raw() const { return m_impl.get(); }

            Tensor() : self(VOID, {}, nullptr) {}

            Tensor(DTYPE dtype, const Shape &shape, const void *data = nullptr)
                    : self(ts_DTYPE(dtype), shape, data) {}

            Tensor(ts_DTYPE dtype, const Shape &shape, const void *data = nullptr)
                    : self(ts_new_Tensor(shape.data(), int32_t(shape.size()), ts_DTYPE(dtype), data)) {
                TS_API_AUTO_CHECK(m_impl != nullptr);
            }

            Shape sizes() const {
                auto shape = ts_Tensor_shape(m_impl.get());
                auto shape_len = ts_Tensor_shape_size(m_impl.get());
                return Shape(shape, shape + shape_len);
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

            self clone() const {
                auto clone_raw = ts_Tensor_clone(m_impl.get());
                TS_API_AUTO_CHECK(clone_raw != nullptr);
                return Tensor(clone_raw);
            }

            void sync_cpu() {
                TS_API_AUTO_CHECK(ts_Tensor_sync_cpu(m_impl.get()));
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

        private:
            Tensor(raw *ptr) : m_impl(pack(ptr)) {}

            static shared_raw pack(raw *ptr) { return shared_raw(ptr, ts_free_Tensor); }

            shared_raw m_impl;
        };

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
                return Tensor(CHAR8, {int32_t(value.length())}, value.c_str());
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
                if (cpu_value.dtype() != INT32) cpu_value = cast(INT32, cpu_value);
                return cast(INT32, cpu_value).data<int32_t>()[0];
            }

            unsigned int to_uint(const Tensor &value) {
                auto cpu_value = value;
                cpu_value.sync_cpu();
                if (cpu_value.dtype() != UINT32) cpu_value = cast(UINT32, cpu_value);
                return cast(UINT32, cpu_value).data<uint32_t>()[0];
            }

            float to_float(const Tensor &value) {
                auto cpu_value = value;
                cpu_value.sync_cpu();
                if (cpu_value.dtype() != FLOAT32) cpu_value = cast(FLOAT32, cpu_value);
                return cast(FLOAT32, cpu_value).data<float>()[0];
            }

            double to_double(const Tensor &value) {
                auto cpu_value = value;
                cpu_value.sync_cpu();
                if (cpu_value.dtype() != FLOAT64) cpu_value = cast(FLOAT64, cpu_value);
                return cast(FLOAT64, cpu_value).data<double>()[0];
            }

            std::string to_string(const Tensor &value) {
                auto cpu_value = value;
                cpu_value.sync_cpu();
                if (cpu_value.dtype() != CHAR8) {
                    cpu_value = cast(CHAR8, cpu_value);
                }
                return std::string(cpu_value.data<char>(), size_t(cpu_value.count()));
            }

            template<typename T>
            inline Tensor build(DTYPE dtype, T &value) {
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
                return cast(dtype, tensor_builder<T>::build(value)).reshape(shape);
            }

            template<typename T>
            inline Tensor build(DTYPE dtype, const Shape &shape, const std::vector<T> &value) {
                return cast(dtype, tensor_builder<T>::build(value)).reshape(shape);
            }

            template<typename T>
            inline Tensor build(DTYPE dtype, const Shape &shape, const T *data) {
                int count = 1;
                for (auto &size : shape) count *= size;
                return cast(dtype, tensor_builder<T>::build(data, size_t(count))).reshape(shape);
            }
        }
    }
}

#endif //TENSORSTACK_API_CPP_TENSOR_H
