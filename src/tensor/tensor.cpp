//
// Created by seeta on 2018/5/25.
//

#include "tensor/tensor.h"

namespace ts {
    tensor::tensor(MemoryController::shared &controller, TYPE type, const std::vector<int> &_shape)
            : tensor(controller, prototype(type, _shape)) {}

    tensor::tensor(const Device &device, TYPE type, const std::vector<int> &_shape)
            : tensor(device, prototype(type, _shape)) {}

    tensor::tensor(TYPE type, const std::vector<int> &_shape)
            : tensor(prototype(type, _shape)) {}

    tensor::tensor(MemoryController::shared &controller, const tensor::prototype &proto)
        : m_memory(controller->alloc(static_cast<size_t>(proto.count() * proto.type_bytes())))
        , m_proto(proto) {}

    tensor::tensor(const Device &device, const tensor::prototype &proto)
            : m_memory(device, static_cast<size_t>(proto.count() * proto.type_bytes()))
            , m_proto(proto) {}

    tensor::tensor(const tensor::prototype &proto)
            : m_memory(static_cast<size_t>(proto.count() * proto.type_bytes()))
            , m_proto(proto) {}
}