//
// Created by kier on 2018/10/30.
//

#include <cassert>
#include "core/tensor_converter.h"

namespace ts {

    TensorConverter::TensorConverter(const MemoryController::shared &controller)
            : m_controller(controller) {}

    TensorConverter::TensorConverter(const MemoryDevice &device)
            : self(std::make_shared<DynamicMemoryController>(device)) {}

    TensorConverter::TensorConverter()
            : self(MemoryDevice(CPU)) {}

    Tensor TensorConverter::from(const std::string &value) const {
        auto length = value.size();
        Tensor tensor(m_controller, CHAR8, Shape({int(length)}));
        memcpy(tensor.data(), tensor.device(), length, value.data(), MemoryDevice(CPU), length);
        return tensor;
    }

    std::string TensorConverter::to_string(const Tensor &value) const {
        assert(value.proto().type() == CHAR8);
        assert(value.proto().sizes().size() == 1);
        auto length = value.proto().sizes()[0];
        if (value.device().type() == CPU) {
            return std::string(value.data<char>(), size_t(length));
        } else {
            std::unique_ptr<char[]> str_data(new char[length]);
            memcpy(str_data.get(), MemoryDevice(CPU), size_t(length), value.data(), value.device(), size_t(length));
            return std::string(str_data.get(), size_t(length));
        }
    }
}