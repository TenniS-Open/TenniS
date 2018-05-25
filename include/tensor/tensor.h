//
// Created by seeta on 2018/5/25.
//

#ifndef TENSORSTACK_TENSOR_TENSOR_H
#define TENSORSTACK_TENSOR_TENSOR_H

#include "mem/memory.h"
#include "type.h"
#include <vector>
#include <stack/stack.h>

namespace ts {
    class tensor {
    public:
        class shape {
            std::string m_layout; ///< NCHW or NHWC
            std::vector<int> m_shape = {};
        };

        const Device &device() const { return m_memory.device(); }

        TYPE type() const { return m_type; }

    private:
        tensor(stack *s);   // allocate memory from stack

        Memory m_memory;
        TYPE m_type;
        shape m_shape;

        friend class stack;
    };
}


#endif //TENSORSTACK_TENSOR_TENSOR_H
