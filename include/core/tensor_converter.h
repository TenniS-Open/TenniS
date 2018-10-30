//
// Created by kier on 2018/10/30.
//

#ifndef TENSORSTACK_CORE_TENSOR_CONVERTER_H
#define TENSORSTACK_CORE_TENSOR_CONVERTER_H

#include "tensor.h"

namespace ts {
    class TensorConverter {
    public:
        using self = TensorConverter;

        TensorConverter();

        explicit TensorConverter(const MemoryDevice &device);

        explicit TensorConverter(const MemoryController::shared &controller);

        Tensor from(const std::string &value) const;

        Tensor from(int8_t value) const;

        Tensor from(int16_t value) const;

        Tensor from(int32_t value) const;

        Tensor from(int64_t value) const;

        Tensor from(uint8_t value) const;

        Tensor from(uint16_t value) const;

        Tensor from(uint32_t value) const;

        Tensor from(uint64_t value) const;

        Tensor from(float value) const;

        Tensor from(double value) const;

        int to_int(const Tensor &value) const;

        unsigned int to_uint(const Tensor &value) const;

        float to_float(const Tensor &value) const;

        double to_double(const Tensor &value) const;

        std::string to_string(const Tensor &value) const;

    private:
        mutable MemoryController::shared m_controller;
    };
}


#endif //TENSORSTACK_CORE_TENSOR_CONVERTER_H
