//
// Created by kier on 2018/10/30.
//

#ifndef TENSORSTACK_CORE_TENSOR_CONVERTER_H
#define TENSORSTACK_CORE_TENSOR_CONVERTER_H

#include "tensor.h"

namespace ts {
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



#endif //TENSORSTACK_CORE_TENSOR_CONVERTER_H
