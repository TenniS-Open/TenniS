//
// Created by seeta on 2018/5/25.
//

#include "core/tensor.h"

namespace ts {
    Tensor::Tensor(MemoryController::shared &controller, TYPE type, const Shape &_shape)
            : Tensor(controller, Prototype(type, _shape)) {}

    Tensor::Tensor(const MemoryDevice &device, TYPE type, const Shape &_shape)
            : Tensor(device, Prototype(type, _shape)) {}

    Tensor::Tensor(TYPE type, const Shape &_shape)
            : Tensor(Prototype(type, _shape)) {}

    Tensor::Tensor(MemoryController::shared &controller, const Tensor::Prototype &proto)
        : m_memory(controller->alloc(static_cast<size_t>(proto.count() * proto.type_bytes())))
        , m_proto(proto) {}

    Tensor::Tensor(const MemoryDevice &device, const Tensor::Prototype &proto)
            : m_memory(device, static_cast<size_t>(proto.count() * proto.type_bytes()))
            , m_proto(proto) {}

    Tensor::Tensor(const Tensor::Prototype &proto)
            : m_memory(static_cast<size_t>(proto.count() * proto.type_bytes()))
            , m_proto(proto) {}

    Tensor Tensor::clone(MemoryController::shared &controller) const {
        Tensor dolly(controller, this->m_proto);
        memcpy(dolly.m_memory, this->m_memory, size_t(this->m_proto.count() * this->m_proto.type_bytes()));
        return std::move(dolly);
    }

    Tensor::shared Tensor::clone_shared(MemoryController::shared &controller) const {
        Tensor::shared dolly = std::make_shared<Tensor>(controller, this->m_proto);
        memcpy(dolly->m_memory, this->m_memory, size_t(this->m_proto.count() * this->m_proto.type_bytes()));
        return std::move(dolly);
    }
}