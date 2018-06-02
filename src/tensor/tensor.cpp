//
// Created by seeta on 2018/5/25.
//

#include "tensor/tensor.h"

namespace ts {
    tensor::tensor(MemoryController::shared &controller, TYPE type, const Shape &_shape)
            : tensor(controller, Prototype(type, _shape)) {}

    tensor::tensor(const Device &device, TYPE type, const Shape &_shape)
            : tensor(device, Prototype(type, _shape)) {}

    tensor::tensor(TYPE type, const Shape &_shape)
            : tensor(Prototype(type, _shape)) {}

    tensor::tensor(MemoryController::shared &controller, const tensor::Prototype &proto)
        : m_memory(controller->alloc(static_cast<size_t>(proto.count() * proto.type_bytes())))
        , m_proto(proto) {}

    tensor::tensor(const Device &device, const tensor::Prototype &proto)
            : m_memory(device, static_cast<size_t>(proto.count() * proto.type_bytes()))
            , m_proto(proto) {}

    tensor::tensor(const tensor::Prototype &proto)
            : m_memory(static_cast<size_t>(proto.count() * proto.type_bytes()))
            , m_proto(proto) {}
}