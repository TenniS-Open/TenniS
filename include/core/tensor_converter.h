//
// Created by kier on 2018/10/30.
//

#ifndef TENSORSTACK_CORE_TENSOR_CONVERTER_H
#define TENSORSTACK_CORE_TENSOR_CONVERTER_H

#include "tensor.h"

namespace ts {
    template <typename T>
    class tensor_builder
    {
        static Tensor build(const T &value) {
            return build(std::vector<T>(1, value));
        }
        static Tensor build(const std::initializer_list<T> &value) {
            return build(std::vector<T>(value.begin(), value.end()));
        }
        static Tensor build(const std::vector<T> &value);
    };

    namespace tensor {
        Tensor from(const std::string &value);

        template<size_t _size>
        inline Tensor from(const char (&value)[_size]) { return from(std::string(value)); }

        inline Tensor from(const char *value) { return from(std::string(value)); }

        Tensor from(int8_t value);

        Tensor from(int16_t value);

        Tensor from(int32_t value);

        Tensor from(int64_t value);

        Tensor from(uint8_t value);

        Tensor from(uint16_t value);

        Tensor from(uint32_t value);

        Tensor from(uint64_t value);

        Tensor from(float value);

        Tensor from(double value);

        int to_int(const Tensor &value);

        unsigned int to_uint(const Tensor &value);

        float to_float(const Tensor &value);

        double to_double(const Tensor &value);

        std::string to_string(const Tensor &value);

        Tensor cast(TYPE type, const Tensor &value);
    }
}

extern template class ts::tensor_builder<ts::type<ts::INT8>::declare>;
extern template class ts::tensor_builder<ts::type<ts::UINT8>::declare>;
extern template class ts::tensor_builder<ts::type<ts::INT16>::declare>;
extern template class ts::tensor_builder<ts::type<ts::UINT16>::declare>;
extern template class ts::tensor_builder<ts::type<ts::INT32>::declare>;
extern template class ts::tensor_builder<ts::type<ts::UINT32>::declare>;
extern template class ts::tensor_builder<ts::type<ts::INT64>::declare>;
extern template class ts::tensor_builder<ts::type<ts::UINT64>::declare>;
extern template class ts::tensor_builder<ts::type<ts::FLOAT32>::declare>;
extern template class ts::tensor_builder<ts::type<ts::FLOAT64>::declare>;



#endif //TENSORSTACK_CORE_TENSOR_CONVERTER_H
