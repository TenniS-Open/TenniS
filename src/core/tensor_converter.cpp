//
// Created by kier on 2018/10/30.
//

#include "core/tensor_converter.h"

namespace ts {

    TensorConverter::TensorConverter(const MemoryController::shared &controller)
        : m_controller(controller) {}

    TensorConverter::TensorConverter(const Device &device)
        : self(std::make_shared<BaseMemoryController>(device)) {}

    TensorConverter::TensorConverter()
        : self(ts::CPU) {}
}