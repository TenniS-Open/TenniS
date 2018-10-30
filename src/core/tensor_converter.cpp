//
// Created by kier on 2018/10/30.
//

#include <cassert>
#include <cstring>
#include "core/tensor_converter.h"

namespace ts {
    namespace tensor {
        Tensor from(const std::string &value) {
            auto length = value.size();
            Tensor tensor(CHAR8, Shape({int(length)}));
            std::memcpy(tensor.data(), value.data(), length);
            return tensor;
        }

        std::string to_string(const Tensor &value) {
            assert(value.proto().type() == CHAR8);
            assert(value.proto().sizes().size() == 1);
            auto cpu_value = value;
            if (cpu_value.device().type() != CPU) {
                auto controller = std::make_shared<DynamicMemoryController>(MemoryDevice(CPU));
                cpu_value = cpu_value.clone(controller);
            }
            auto length = cpu_value.proto().sizes()[0];
            return std::string(cpu_value.data<char>(), size_t(length));
        }
    }
}