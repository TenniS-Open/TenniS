//
// Created by seeta on 2018/5/25.
//

#ifndef TENSORSTACK_RUNTIME_STACK_H
#define TENSORSTACK_RUNTIME_STACK_H

#include "core/device.h"
#include "core/controller.h"
#include "core/dtype.h"
#include "core/tensor.h"
#include "global/hard_converter.h"

#include <vector>
#include <deque>
#include <stack>
#include <cassert>

namespace ts {

    class Stack {
    public:
        using self = Stack;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        explicit Stack(const MemoryDevice &device)
                : Stack(device, std::make_shared<DynamicMemoryController>(device)) {}

        explicit Stack(const MemoryDevice &device, const MemoryController::shared &controller)
                : m_device(device), m_controller(controller) {}

        // return new Tensor, but not in stack
        Tensor make(DTYPE dtype, const Shape &shape) {
            return Tensor(m_controller, dtype, shape);
        }

        Tensor *push(DTYPE dtype, const Shape &shape) {
            return this->push(this->make(dtype, shape));
        }

        Tensor *push(const Tensor::Prototype &proto) {
            return this->push(proto.dtype(), proto.sizes());
        }

        Tensor *push(const Tensor &tensor) {
            // TODO: remove this check, supporting cross device computing
            assert(tensor.device() == this->m_device);
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

        Tensor *clone(int i) {
            auto &tensor = *this->index(i);
            auto &proto = tensor.proto();
            auto dolly = this->push(proto.dtype(), proto.sizes());
            auto copy_converter = this->converter();
            copy_converter(m_device.id(), dolly->data(),
                           m_device.id(), tensor.data(),
                           size_t(tensor.proto().count() * tensor.proto().type_bytes()));
            return dolly;
        }

        // if i >= 0, then i is bottom_up_index; else if i < 0, the i is top_down_index
        Tensor *index(int i) { return &this->m_stack.at(relative2absolute(i)); }

        const Tensor *index(int i) const { return &this->m_stack.at(relative2absolute(i)); }

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
            // assert(end_it - beg_it >= 0);
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
                assert(this->m_converter != nullptr);
            }
            return this->m_converter;
        }

    private:
        size_t relative2absolute(int i) const {
            return i >= 0 ? m_base + i : m_stack.size() + i;
        }

        Device m_device;                          ///< running tensor device, compute on it
        MemoryController::shared m_controller;    ///< tensor memory backend
        std::deque<Tensor> m_stack;               ///< saving all tensor
        size_t m_base = 0;                        ///< the running control base
        std::stack<size_t> m_base_stack;          ///< save each call base

        mutable HardConverter::function m_converter = nullptr;    ///< convert memory in stack
    };
}


#endif //TENSORSTACK_RUNTIME_STACK_H
