//
// Created by kier on 2018/10/30.
//

#include "core/tensor_converter.h"

namespace ts {

    TensorConverter::TensorConverter(const MemoryController::shared &controller)
            : m_controller(controller) {}

    TensorConverter::TensorConverter(const MemoryDevice &device)
            : self(std::make_shared<BaseMemoryController>(device)) {}

    TensorConverter::TensorConverter()
            : self(MemoryDevice(ts::CPU)) {}

    Tensor TensorConverter::from(const std::string &value) const {
        auto length = value.size();
        Tensor tensor(m_controller, CHAR8, Shape({int(length)}));
        memcpy(tensor.data(), tensor.device(), length, value.data(), MemoryDevice(ts::CPU), length);
        return tensor;
    }

    const TensorConverter &tensor_converter() {
        static TensorConverter cpu_converter;
        return cpu_converter;
    }
}