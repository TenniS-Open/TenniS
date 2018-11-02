//
// Created by seeta on 2018/5/25.
//

#include <core/tensor.h>
#include <cassert>

#include "core/tensor.h"

namespace ts {
    Tensor::Tensor(MemoryController::shared controller, DTYPE dtype, const Shape &_shape)
            : Tensor(controller, Prototype(dtype, _shape)) {}

    Tensor::Tensor(const MemoryDevice &device, DTYPE dtype, const Shape &_shape)
            : Tensor(device, Prototype(dtype, _shape)) {}

    Tensor::Tensor(DTYPE dtype, const Shape &_shape)
            : Tensor(Prototype(dtype, _shape)) {}

    Tensor::Tensor(MemoryController::shared controller, const Tensor::Prototype &proto)
            : m_memory(controller->alloc(static_cast<size_t>(proto.count() * proto.type_bytes())))
            , m_proto(proto) {}

    Tensor::Tensor(const MemoryDevice &device, const Tensor::Prototype &proto)
            : m_memory(device, static_cast<size_t>(proto.count() * proto.type_bytes()))
            , m_proto(proto) {}

    Tensor::Tensor(const Tensor::Prototype &proto)
            : m_memory(static_cast<size_t>(proto.count() * proto.type_bytes()))
            , m_proto(proto) {}

    Tensor::Tensor(const Memory &memory, const Tensor::Prototype &proto)
            : m_memory(memory)
            , m_proto(proto) {}

    Tensor Tensor::clone(MemoryController::shared controller) const {
        auto fields = this->unpack();
        for (auto &value : fields) {
            Tensor dolly(std::move(controller), value.m_proto);
            memcpy(dolly.m_memory, value.m_memory, size_t(value.m_proto.count() * value.m_proto.type_bytes()));
            value = dolly;
        }
        Tensor dolly;
        dolly.pack(fields);
        return std::move(dolly);
    }

    Tensor::shared Tensor::clone_shared(MemoryController::shared controller) const {
        return std::make_shared<Tensor>(this->clone(std::move(controller)));
    }

    Tensor::Tensor()
            : Tensor(VOID, {}) {
    }

    bool Tensor::empty() const {
        return this->count() == 0;
    }

    Tensor Tensor::reshape(const Shape &shape) const {
        Prototype proto(this->dtype(), shape);
        assert(proto.count() == this->count());
        Tensor t = *this;
        t.m_proto = proto;
        return t;
    }

    void Tensor::pack(const std::vector<Tensor::self> &fields) {
        if (fields.empty()) {
            this->m_memory = Memory();
            this->m_proto = Prototype();
        }
        this->m_memory = fields[0].m_memory;
        this->m_proto = fields[0].m_proto;
        this->m_fields = std::make_shared<std::vector<self>>(fields.begin() + 1, fields.end());
    }

    std::vector<Tensor::self> Tensor::unpack() const {
        std::vector<Tensor::self> fields(1);
        fields[0].m_memory = this->m_memory;
        fields[0].m_proto = this->m_proto;
        if (this->m_fields != nullptr) {
            fields.insert(fields.end(), this->m_fields->begin(), this->m_fields->end());
        }
        return std::move(fields);
    }

    Tensor Tensor::field(size_t offset) const {
        if (offset == 0) {
            return Tensor(m_memory, m_proto);
        }
        if (m_fields == nullptr || offset - 1 >= m_fields->size()) {
            throw Exception(std::string("Tensor offset output range error. Access index ") + std::to_string(offset) +
                            " in range(" + std::to_string(fields_count()) + ")");
        }
        return m_fields->at(offset - 1);
    }

    void Tensor::field(size_t offset, const Tensor::self &value) {
        if (offset == 0) {
            this->m_memory = value.m_memory;
            this->m_proto = value.m_proto;
            return;
        }
        if (m_fields == nullptr || offset - 1 >= m_fields->size()) {
            throw Exception(std::string("Tensor offset output range error. Access index ") + std::to_string(offset) +
                            " in range(" + std::to_string(fields_count()) + ")");
        }
        m_fields->at(offset - 1) = value;
    }

    size_t Tensor::fields_count() const {
        return m_fields == nullptr ? 1 : 1 + m_fields->size();
    }
}