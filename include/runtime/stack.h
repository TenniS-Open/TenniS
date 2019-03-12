//
// Created by kier on 2018/5/25.
//

#ifndef TENSORSTACK_RUNTIME_STACK_H
#define TENSORSTACK_RUNTIME_STACK_H

#include "core/device.h"
#include "core/sync/sync_controller.h"
#include "core/dtype.h"
#include "core/tensor.h"
#include "global/hard_converter.h"
#include "utils/assert.h"

#include <vector>
#include <deque>
#include <stack>


namespace ts {

    class TS_DEBUG_API Stack {
    public:
        using self = Stack;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        explicit Stack(const MemoryDevice &device)
                : Stack(device, DynamicSyncMemoryController::Make(device)) {}

        explicit Stack(const MemoryDevice &device, bool need_lock)
                : Stack(device, DynamicSyncMemoryController::Make(device, need_lock)) {}

        explicit Stack(const MemoryDevice &device, const SyncMemoryController::shared &controller)
                : m_device(device), m_controller(controller) {}

        // return new Tensor, but not in stack
        Tensor make(DTYPE dtype, const Shape &shape) {
            return Tensor(m_controller, dtype, shape);
        }

        Tensor make(DTYPE dtype, const Shape &shape, const MemoryDevice &device) {
            return Tensor(m_controller, dtype, shape, device);
        }

        Tensor *push(DTYPE dtype, const Shape &shape) {
            return this->push(this->make(dtype, shape));
        }

        Tensor *push(DTYPE dtype, const Shape &shape, const MemoryDevice &device) {
            return this->push(this->make(dtype, shape, device));
        }

        Tensor *push(const Tensor::Prototype &proto) {
            return this->push(proto.dtype(), proto.sizes());
        }

        Tensor *push(const Tensor::Prototype &proto, const MemoryDevice &device) {
            return this->push(proto.dtype(), proto.sizes(), device);
        }

        Tensor *push(const Tensor &tensor) {
            // removed this check, supporting cross device computing
            // TS_AUTO_CHECK(tensor.device() == this->m_device);
            this->m_stack.push_back(tensor);
            return &this->m_stack.back();
        }

        Tensor *push(int i) {
            return this->push(*this->index(i));
        }

        /**
         * clone_push means push an clone of tensor
         * @param tensor param to clone
         * @return return cloned tensor
         */
        Tensor *clone_push(const Tensor &tensor) {
            return this->push(tensor.clone(this->m_controller));
        }

        Tensor *clone_push(const Tensor &tensor, const MemoryDevice &device) {
            return this->push(tensor.clone(this->m_controller, device));
        }

        Tensor *clone(int i) {
            auto tensor = *this->index(i);
            return clone_push(tensor);
        }

        Tensor *clone(int i, const MemoryDevice &device) {
            auto tensor = *this->index(i);
            return clone_push(tensor, device);
        }

        // if i >= 0, then i is bottom_up_index; else if i < 0, the i is top_down_index
        Tensor *index(int i) { return &this->m_stack.at(relative2absolute(i)); }

        const Tensor *index(int i) const { return &this->m_stack.at(relative2absolute(i)); }

        Tensor &operator[](int i) { return *index(i); }

        const Tensor &operator[](int i) const { return *index(i); }

        Tensor &operator[](size_t i) { return *index(int(i)); }

        const Tensor &operator[](size_t i) const { return *index(int(i)); }

        Tensor *top() { return &this->m_stack.back(); }

        const Tensor *top() const { return &this->m_stack.back(); }

        void pop(size_t pop_size) {
            auto it_end = this->m_stack.end();
            auto it_beg = it_end - pop_size;
            this->m_stack.erase(it_beg, it_end);
        }

        void pop() { this->m_stack.pop_back(); }

        size_t size() const { return m_stack.size() - m_base; }

        void erase(int i) {
            auto it = this->m_stack.begin() + relative2absolute(i);
            this->m_stack.erase(it);
        }

        // remove [beg, end)
        void erase(int beg, int end) {
            auto beg_it = this->m_stack.begin() + relative2absolute(beg);
            auto end_it = this->m_stack.begin() + relative2absolute(end);
            // TS_AUTO_CHECK(end_it - beg_it >= 0);
            this->m_stack.erase(beg_it, end_it);
        }

        void clear() {this->erase(0, static_cast<int>(this->size()));}

        size_t base() const { return m_base; }

        // return old base
        size_t rebase(size_t new_base) {
            std::swap(m_base, new_base);
            return new_base;
        }

        void push_base(int i) {
            this->m_base_stack.push(this->rebase(relative2absolute(i)));
        }
        void pop_base() {
            if (this->m_base_stack.empty()){
                this->rebase(0);
            } else {
                this->rebase(this->m_base_stack.top());
                this->m_base_stack.pop();
            }
        }

        HardConverter::function converter() const {
            if (this->m_converter == nullptr) {
                this->m_converter = HardConverter::Query(m_device.type(), m_device.type());
                TS_AUTO_CHECK(this->m_converter != nullptr);
            }
            return this->m_converter;
        }

    private:
        size_t relative2absolute(int i) const {
            return i >= 0 ? m_base + i : m_stack.size() + i;
        }

        Device m_device;                          ///< running tensor device, compute on it
        SyncMemoryController::shared m_controller;    ///< tensor memory backend
        std::deque<Tensor> m_stack;               ///< saving all tensor
        size_t m_base = 0;                        ///< the running control base
        std::stack<size_t> m_base_stack;          ///< save each call base

        mutable HardConverter::function m_converter = nullptr;    ///< convert memory in stack
    };
}


#endif //TENSORSTACK_RUNTIME_STACK_H
