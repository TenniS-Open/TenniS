//
// Created by kier on 2018/10/30.
//

#ifndef TENSORSTACK_CORE_TENSOR_CONVERTER_H
#define TENSORSTACK_CORE_TENSOR_CONVERTER_H

#include "tensor.h"

namespace ts {
    template <typename T>
    class TS_DEBUG_API tensor_builder
    {
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
        static Tensor build(const T *data, size_t count);
    };

    template <>
    inline Tensor tensor_builder<bool>::build(const std::vector<bool> &value) {
        Tensor t(BOOLEAN, {int(value.size())});
        using type = ts::dtype<ts::BOOLEAN>::declare;
        auto out = t.data<type>();
        for (int i = 0; i < t.count(); ++i) {
            out[i] = type(value[i] ? 1 : 0);
        }
        return t;
    }

    template <>
    inline Tensor tensor_builder<bool>::build(const bool *data, size_t count) {
        Tensor t(BOOLEAN, {int(count)});
        using type = ts::dtype<ts::BOOLEAN>::declare;
        auto out = t.data<type>();
        for (int i = 0; i < t.count(); ++i) {
            out[i] = type(data[i] ? 1 : 0);
        }
        return t;
    }

    namespace tensor {
        TS_DEBUG_API Tensor from(const std::string &value);

        template<size_t _size>
        inline Tensor from(const char (&value)[_size]) { return from(std::string(value)); }

        inline Tensor from(const char *value) { return from(std::string(value)); }

        template <typename T>
        Tensor from(const T value) { return tensor_builder<T>::build(value); }

        template <typename T>
        Tensor from(const std::initializer_list<T> &value) { return tensor_builder<T>::build(value); }

        template <typename T>
        Tensor from(const std::vector<T> &value) { return tensor_builder<T>::build(value); }

        TS_DEBUG_API int to_int(const Tensor &value);

        TS_DEBUG_API unsigned int to_uint(const Tensor &value);

        TS_DEBUG_API float to_float(const Tensor &value);

        TS_DEBUG_API double to_double(const Tensor &value);

        TS_DEBUG_API std::string to_string(const Tensor &value);

        TS_DEBUG_API bool to_bool(const Tensor &value);

        TS_DEBUG_API Tensor cast(DTYPE dtype, const Tensor &value);

        TS_DEBUG_API Tensor clone(DTYPE dtype, const Tensor &value);

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
            return cast(dtype, tensor_builder<T>::build(data, count)).reshape(shape);
        }

        TS_DEBUG_API bool support(DTYPE dtype);

        TS_DEBUG_API Tensor load(StreamReader &stream);

        TS_DEBUG_API Tensor load(const std::string &filename);

        TS_DEBUG_API void save(StreamWriter &stream, const Tensor &tensor);

        TS_DEBUG_API void save(const std::string &filename, const Tensor &tensor);
    }
}

// extern template class ts::tensor_builder<ts::dtype<ts::CHAR8>::declare>;
extern template class ts::tensor_builder<ts::dtype<ts::INT8>::declare>;
extern template class ts::tensor_builder<ts::dtype<ts::UINT8>::declare>;
extern template class ts::tensor_builder<ts::dtype<ts::INT16>::declare>;
extern template class ts::tensor_builder<ts::dtype<ts::UINT16>::declare>;
extern template class ts::tensor_builder<ts::dtype<ts::INT32>::declare>;
extern template class ts::tensor_builder<ts::dtype<ts::UINT32>::declare>;
extern template class ts::tensor_builder<ts::dtype<ts::INT64>::declare>;
extern template class ts::tensor_builder<ts::dtype<ts::UINT64>::declare>;
extern template class ts::tensor_builder<ts::dtype<ts::FLOAT32>::declare>;
extern template class ts::tensor_builder<ts::dtype<ts::FLOAT64>::declare>;



#endif //TENSORSTACK_CORE_TENSOR_CONVERTER_H
