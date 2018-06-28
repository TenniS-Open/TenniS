//
// Created by seeta on 2018/5/25.
//

#ifndef TENSORSTACK_STACK_STACK_H
#define TENSORSTACK_STACK_STACK_H

#include <global/device.h>
#include <mem/controller.h>
#include <vector>
#include <tensor/type.h>
#include <tensor/tensor.h>
#include <deque>

namespace ts {

    class Stack {
    public:
        explicit Stack(const Device &device)
                : Stack(device, std::make_shared<BaseMemoryController>(device)) {}

        explicit Stack(const Device &device, const MemoryController::shared &controller)
                : m_device(device), m_controller(controller) {}

        // return new Tensor, but not in stack
        Tensor make(TYPE type, const Shape &shape) {
            return Tensor(m_controller, type, shape);
        }

        Tensor *push(TYPE type, const Shape &shape) {
            return this->push(this->make(type, shape));
        }

        Tensor *push(const Tensor &tensor) {
            this->m_stack.push_back(tensor);
            return &this->m_stack.back();
        }

        Tensor *push(int i) {
            return this->push(*this->index(i));
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

        size_t base() const { return m_base; }

        // return old base
        size_t rebase(size_t new_base) {
            std::swap(m_base, new_base);
            return new_base;
        }

    private:
        size_t relative2absolute(int i) const {
            return i >= 0 ? m_base + i : m_stack.size() + i;
        }

        Device m_device;                          ///< running tensor device, compute on it
        MemoryController::shared m_controller;    ///< tensor memory backend
        std::deque<Tensor> m_stack;               ///< saving all tensor
        size_t m_base = 0;                        ///< the running control base
    };
}


#endif //TENSORSTACK_STACK_STACK_H
