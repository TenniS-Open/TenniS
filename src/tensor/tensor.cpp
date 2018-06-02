//
// Created by seeta on 2018/5/25.
//

#include "tensor/tensor.h"

namespace ts {
    Tensor::Tensor(MemoryController::shared &controller, TYPE type, const Shape &_shape)
            : Tensor(controller, Prototype(type, _shape)) {}

    Tensor::Tensor(const Device &device, TYPE type, const Shape &_shape)
            : Tensor(device, Prototype(type, _shape)) {}

    Tensor::Tensor(TYPE type, const Shape &_shape)
            : Tensor(Prototype(type, _shape)) {}

    Tensor::Tensor(MemoryController::shared &controller, const Tensor::Prototype &proto)
        : m_memory(controller->alloc(static_cast<size_t>(proto.count() * proto.type_bytes())))
        , m_proto(proto) {}

    Tensor::Tensor(const Device &device, const Tensor::Prototype &proto)
            : m_memory(device, static_cast<size_t>(proto.count() * proto.type_bytes()))
            , m_proto(proto) {}

    Tensor::Tensor(const Tensor::Prototype &proto)
            : m_memory(static_cast<size_t>(proto.count() * proto.type_bytes()))
            , m_proto(proto) {}
}