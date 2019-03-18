//
// Created by kier on 2018/5/25.
//

#include <core/tensor.h>

#include <utility>

#include <utils/ctxmgr.h>
#include "core/tensor.h"
#include "utils/assert.h"

#include <numeric>

namespace ts {
    Tensor::Tensor(MemoryController::shared controller, DTYPE dtype, const Shape &_shape)
            : Tensor(controller, Prototype(dtype, _shape)) {}

    Tensor::Tensor(SyncMemoryController::shared controller, DTYPE dtype, const Shape &_shape)
            : Tensor(controller, Prototype(dtype, _shape)) {}

    Tensor::Tensor(SyncMemoryController::shared controller, DTYPE dtype, const Shape &_shape, const MemoryDevice &device)
            : Tensor(controller, Prototype(dtype, _shape), device) {}

    Tensor::Tensor(const MemoryDevice &device, DTYPE dtype, const Shape &_shape)
            : Tensor(device, Prototype(dtype, _shape)) {}

    Tensor::Tensor(DTYPE dtype, const Shape &_shape)
            : Tensor(Prototype(dtype, _shape)) {}

    Tensor::Tensor(MemoryController::shared controller, const Tensor::Prototype &proto)
            : m_memory(controller->alloc(static_cast<size_t>(proto.count() * proto.type_bytes())))
            , m_proto(proto) {}

    Tensor::Tensor(SyncMemoryController::shared controller, const Tensor::Prototype &proto)
            : m_memory(controller->alloc(static_cast<size_t>(proto.count() * proto.type_bytes())))
            , m_proto(proto) {}

    Tensor::Tensor(SyncMemoryController::shared controller, const Tensor::Prototype &proto, const MemoryDevice &device)
            : m_memory(controller->alloc(device, static_cast<size_t>(proto.count() * proto.type_bytes())))
            , m_proto(proto) {}

    Tensor::Tensor(const MemoryDevice &device, const Tensor::Prototype &proto)
            : m_memory(make_smart<TensorMemory>(device, static_cast<size_t>(proto.count() * proto.type_bytes())))
            , m_proto(proto) {}

    Tensor::Tensor(const Tensor::Prototype &proto)
            : m_memory(static_cast<size_t>(proto.count() * proto.type_bytes()))
            , m_proto(proto) {}

    Tensor::Tensor(const Memory &memory, const Tensor::Prototype &proto)
            : m_memory(memory)
            , m_proto(proto) {}

    Tensor::Tensor(const SyncMemory &memory, const Tensor::Prototype &proto)
            : m_memory(memory)
            , m_proto(proto) {}

    Tensor::Tensor(const Smart<TensorMemory> &memory, const Tensor::Prototype &proto)
            : m_memory(memory)
            , m_proto(proto) {}

    Tensor Tensor::clone() const {
        std::shared_ptr<MemoryController> controller = std::make_shared<DynamicMemoryController>(this->device());
        return clone(controller);
    }

    Tensor Tensor::clone(MemoryController::shared controller) const {
        auto fields = this->unpack();
        for (auto &value : fields) {
            Tensor dolly(std::move(controller), value.m_proto);
            memcpy(*dolly.m_memory, *value.m_memory, size_t(value.m_proto.count() * value.m_proto.type_bytes()));
            value = dolly;
        }
        Tensor dolly;
        dolly.pack(fields);
        return std::move(dolly);
    }

    Tensor Tensor::clone(SyncMemoryController::shared controller) const {
        auto fields = this->unpack();
        for (auto &value : fields) {
            Tensor dolly(std::move(controller), value.m_proto);
            memcpy(*dolly.m_memory, *value.m_memory, size_t(value.m_proto.count() * value.m_proto.type_bytes()));
            value = dolly;
        }
        Tensor dolly;
        dolly.pack(fields);
        return std::move(dolly);
    }

    Tensor Tensor::clone(SyncMemoryController::shared controller, const MemoryDevice &device) const {
        auto fields = this->unpack();
        for (auto &value : fields) {
            Tensor dolly(std::move(controller), value.m_proto, device);
            memcpy(*dolly.m_memory, *value.m_memory, size_t(value.m_proto.count() * value.m_proto.type_bytes()));
            value = dolly;
        }
        Tensor dolly;
        dolly.pack(fields);
        return std::move(dolly);
    }

    Tensor::shared Tensor::clone_shared(MemoryController::shared controller) const {
        return std::make_shared<Tensor>(this->clone(std::move(controller)));
    }

    Tensor::shared Tensor::clone_shared() const {
        return std::make_shared<Tensor>(this->clone());
    }

    Tensor::shared Tensor::clone_shared(SyncMemoryController::shared controller) const {
        return std::make_shared<Tensor>(this->clone(std::move(controller)));
    }

    Tensor::shared Tensor::clone_shared(SyncMemoryController::shared controller, const MemoryDevice &device) const {
        return std::make_shared<Tensor>(this->clone(std::move(controller), device));
    }

    Tensor::Tensor()
            : Tensor(VOID, {}) {
    }

    bool Tensor::empty() const {
        return this->count() == 0;
    }

    Tensor Tensor::reshape(const Shape &shape) const {
        auto fixed_shape = shape;
        int64_t fixed_index = -1;
        for (size_t i = 0; i < fixed_shape.size(); ++i) {
            if (fixed_shape[i] < 0) {
                if (fixed_index >= 0) {
                    TS_LOG_ERROR << "Can not reshape " << to_string(this->sizes()) << " to " << to_string(shape)
                                 << eject;
                }
                fixed_shape[i] = -1;
                fixed_index = int64_t(i);
            }
        }
        if (fixed_index >= 0) {
            auto up = std::accumulate(this->sizes().begin(), this->sizes().end(), 1, std::multiplies<int>());
            auto down = std::accumulate(fixed_shape.begin(), fixed_shape.end(), 1, std::multiplies<int>());
            fixed_shape[fixed_index] = up / -down;
        }

        Prototype proto(this->dtype(), fixed_shape);
        if (proto.count() != this->count()) {
            TS_LOG_ERROR << "Can not reshape " << to_string(this->sizes()) << " to " << to_string(shape) << eject;
        }
        Tensor t = *this;
        t.m_proto = std::move(proto);
        return std::move(t);
    }

    void Tensor::pack(const std::vector<Tensor::self> &fields) {
        if (fields.empty()) {
            this->m_memory = make_smart<TensorMemory>();
            this->m_proto = Prototype();
            return;
        }
        this->m_memory = fields[0].m_memory;
        this->m_proto = fields[0].m_proto;
        if (fields.size() > 1) {
            this->m_fields = std::vector<self>(fields.begin() + 1, fields.end());
        } else {
            this->m_fields.clear();
        }
    }

    std::vector<Tensor::self> Tensor::unpack() const {
        std::vector<Tensor::self> fields(1);
        fields[0].m_memory = this->m_memory;
        fields[0].m_proto = this->m_proto;
        if (!this->m_fields.empty()) {
            fields.insert(fields.end(), this->m_fields.begin(), this->m_fields.end());
        }
        return std::move(fields);
    }

    Tensor Tensor::field(size_t offset) const {
        if (offset == 0) {
            return Tensor(m_memory, m_proto);
        }
        if (offset - 1 >= m_fields.size()) {
            TS_LOG_ERROR << "Tensor offset output range error. Access index " << offset << " in range("
                         << fields_count() << ")" << eject;
        }
        return m_fields.at(offset - 1);
    }

    void Tensor::field(size_t offset, const Tensor::self &value) {
        if (offset == 0) {
            this->m_memory = value.m_memory;
            this->m_proto = value.m_proto;
            return;
        }
        if (offset - 1 >= m_fields.size()) {
            TS_LOG_ERROR << "Tensor offset output range error. Access index " << offset << " in range("
                         << fields_count() << ")" << eject;
        }
        m_fields.at(offset - 1) = value;
    }

    size_t Tensor::fields_count() const {
        return m_fields.empty() ? 1 : 1 + m_fields.size();
    }

    bool Tensor::packed() const {
        return !m_fields.empty();
    }

    static size_t serialize_prototype_memory(StreamWriter &stream,
                                             const Tensor::Prototype &proto, const Memory &memory) {
        size_t writen_size = 0;
        // 1. write prototype
        // 1.1 write dtype
        writen_size += binio::write<uint8_t>(stream, proto.dtype());
        // 1.2 write size
        writen_size += binio::write<uint32_t>(stream, uint32_t(proto.sizes().size()));
        for (auto &size : proto.sizes()) {
            writen_size += binio::write<uint32_t>(stream, uint32_t(size));
        }
        Memory cpu_memory;
        if (memory.device().type() == ts::CPU) {
            cpu_memory = memory;
        } else {
            cpu_memory = Memory(memory.size());
            memcpy(cpu_memory, memory);
        }
        // 2. write memory
        writen_size += binio::write<char>(stream, cpu_memory.data<char>(), size_t(proto.count()) * proto.type_bytes());
        return writen_size;
    }

    static size_t externalize_prototype_memory(StreamReader &stream,
                                               Tensor::Prototype &proto, Memory &memory) {
        std::unique_ptr<MemoryController> may_controller;
        auto controller =  ctx::ptr<MemoryController>();
        if (controller == nullptr) {
            may_controller.reset(new DynamicMemoryController(memory.device()));
            controller = may_controller.get();
        }
        size_t read_size = 0;
        // 1. read prototype
        DTYPE dtype;
        Shape shape;
        // 1.1 read dtype
        uint8_t dtype_buffer;
        read_size += binio::read<uint8_t >(stream, dtype_buffer);
        dtype = DTYPE(dtype_buffer);
        TS_AUTO_CHECK(dtype >= VOID && dtype <= UNKNOWN128);
        // 1.2 read sizes
        uint32_t size_buffer;
        read_size += binio::read<uint32_t>(stream, size_buffer);
        shape.resize(size_buffer);
        for (size_t i = 0; i < shape.size(); ++i) {
            read_size += binio::read<uint32_t>(stream, size_buffer);
            shape[i] = size_buffer;
        }
        // 1.x set proto
        proto = Tensor::Prototype(dtype, shape);

        // 2. read memory
        memory = controller->alloc(size_t(proto.count()) * proto.type_bytes());
        read_size += binio::read<char>(stream, memory.data<char>(), memory.size());
        return read_size;
    }

    size_t Tensor::serialize(StreamWriter &stream) const {
        size_t writen_size = 0;
        writen_size += binio::write<uint32_t>(stream, uint32_t(this->fields_count()));
        for (auto &tensor : this->unpack()) {
            auto cpu_memory = tensor.m_memory->sync(MemoryDevice(CPU));
            writen_size += serialize_prototype_memory(stream, tensor.m_proto, cpu_memory);
        }
        return writen_size;
    }

    size_t Tensor::externalize(StreamReader &stream) {
        size_t read_size = 0;
        uint32_t size_buffer;
        read_size += binio::read<uint32_t>(stream, size_buffer);
        std::vector<Tensor> fields(size_buffer);
        for (auto &tensor : fields) {
            Memory cpu_memory;
            read_size += externalize_prototype_memory(stream, tensor.m_proto, cpu_memory);
            tensor.m_memory = SyncMemory(cpu_memory);
        }
        this->pack(fields);
        return read_size;
    }

    Tensor::Tensor(Tensor::self &&other) TS_NOEXCEPT {
        this->operator=(std::forward<self>(other));
    }

    Tensor::self &Tensor::operator=(Tensor::self &&other) TS_NOEXCEPT {
        this->m_proto = std::move(other.m_proto);
        this->m_memory = std::move(other.m_memory);
        this->m_fields = std::move(other.m_fields);
        return *this;
    }

    Tensor Tensor::view(const MemoryDevice &device) const {
        Tensor view_tensor;
        view_tensor.m_memory = TensorMemory(m_memory->sync(device), false);
        view_tensor.m_proto = m_proto;

        if (!m_fields.empty()) {
            std::vector<self> view_fields(m_fields.size());
            for (size_t i = 0; i < m_fields.size(); ++i) {
                view_fields[i] = m_fields.at(i).view(device);
            }

            view_tensor.m_fields = std::vector<self>(std::move(view_fields));
        }

        return view_tensor;
    }

    Tensor Tensor::weak() const {
        Tensor weak_tensor;
        weak_tensor.m_memory = m_memory.weak();
        weak_tensor.m_proto = m_proto;

        if (!m_fields.empty()) {
            std::vector<self> weak_fields(m_fields.size());
            for (size_t i = 0; i < m_fields.size(); ++i) {
                weak_fields[i] = m_fields.at(i).weak();
            }

            weak_tensor.m_fields = std::vector<self>(std::move(weak_fields));
        }

        return weak_tensor;
    }

    bool Tensor::has_shape(const Shape &shape) const {
        auto &this_shape = this->sizes();
        if (this_shape.size() != shape.size()) return false;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] >= 0 && this_shape[i] != shape[i]) return false;
        }
        return true;
    }

    bool Tensor::has_empty_shape() const {
        return this->sizes().empty();
    }

    static inline std::vector<int> flatten_shape(const Tensor &x, int m_dim) {
        auto need_size = size_t(m_dim + 1);
        auto x_size = x.sizes().size();
        if (need_size < x_size) {
            auto &size = x.sizes();
            std::vector<int> shape(size.begin(), size.begin() + need_size);
            shape.back() = std::accumulate(size.begin() + m_dim, size.end(), 1, std::multiplies<int>());
            return std::move(shape);
        } else if (need_size > x_size) {
            std::vector<int> ones(need_size - x_size, 1);
            auto shape = x.sizes();
            shape.insert(shape.end(), ones.begin(), ones.end());
            return std::move(shape);
        } else {
            return x.sizes();
        }
    }

    Tensor Tensor::flatten(int dim) const {
        auto fixed_shape = flatten_shape(*this, dim < 0 ? 0 : dim);
        Prototype proto(this->dtype(), fixed_shape);
        Tensor t = *this;
        t.m_proto = proto;
        return t;
    }

#define FAIL_SIZE(n) (this_shape.size() != (n))
#define FAIL_ARG(i) (arg##i >= 0 && this_shape[i] != arg##i)

    bool Tensor::has_shape(int arg0) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(1) || FAIL_ARG(0));
    }

    bool Tensor::has_shape(int arg0, int arg1) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(2) || FAIL_ARG(0) || FAIL_ARG(1));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(3) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2, int arg3) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(4) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2) || FAIL_ARG(3));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2, int arg3, int arg4) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(5) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2) || FAIL_ARG(3) || FAIL_ARG(4));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2, int arg3, int arg4,
                           int arg5) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(6) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2) || FAIL_ARG(3) || FAIL_ARG(4) ||
                 FAIL_ARG(5));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2, int arg3, int arg4,
                           int arg5, int arg6) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(7) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2) || FAIL_ARG(3) || FAIL_ARG(4) ||
                 FAIL_ARG(5) || FAIL_ARG(6));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2, int arg3, int arg4,
                           int arg5, int arg6, int arg7) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(8) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2) || FAIL_ARG(3) || FAIL_ARG(4) ||
                 FAIL_ARG(5) || FAIL_ARG(6) || FAIL_ARG(7));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2, int arg3, int arg4,
                           int arg5, int arg6, int arg7, int arg8) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(9) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2) || FAIL_ARG(3) || FAIL_ARG(4) ||
                 FAIL_ARG(5) || FAIL_ARG(6) || FAIL_ARG(7) || FAIL_ARG(8));
    }

    bool Tensor::has_shape(int arg0, int arg1, int arg2, int arg3, int arg4,
                           int arg5, int arg6, int arg7, int arg8, int arg9) const {
        auto &this_shape = this->sizes();
        return !(FAIL_SIZE(10) || FAIL_ARG(0) || FAIL_ARG(1) || FAIL_ARG(2) || FAIL_ARG(3) || FAIL_ARG(4) ||
                 FAIL_ARG(5) || FAIL_ARG(6) || FAIL_ARG(7) || FAIL_ARG(8) || FAIL_ARG(9));
    }

#undef FAIL_ARG
#undef FAIL_SIZE
}