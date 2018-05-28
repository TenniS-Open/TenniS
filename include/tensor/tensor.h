//
// Created by seeta on 2018/5/25.
//

#ifndef TENSORSTACK_TENSOR_TENSOR_H
#define TENSORSTACK_TENSOR_TENSOR_H

#include "mem/memory.h"
#include "type.h"
#include <vector>
#include <mem/controller.h>

namespace ts {
    class tensor {
    public:
        class shape {
        public:
            shape() {}
            shape(const std::vector<int> &sizes) : m_sizes(sizes) {}
            shape(const std::vector<int> &sizes, const std::string &layout) : m_sizes(sizes), m_layout(layout) {}

            size_t dims() const {return m_sizes.size();}
            const std::vector<int> &sizes() const {return m_sizes;}
            int operator[](size_t axis) const {return m_sizes[axis];}
            const std::string &layout() const {return m_layout;}

        private:
            std::vector<int> m_sizes = {};
            std::string m_layout; ///< NCHW or NHWC
        };
        tensor(MemoryController::shared &controller);   // allocate memory from controller
        tensor(const Device &device, const std::vector<int> &shape) : m_memory(device, (size_t)(count(shape))), m_shape(shape) {};   // allocate memory from device

        const Device &device() const { return m_memory.device(); }

        TYPE type() const { return m_type; }

        int count() const;
        static int count(const std::vector<int> &shape);

    private:

        Memory m_memory;
        TYPE m_type;
        shape m_shape;

        friend class stack;
    };
}


#endif //TENSORSTACK_TENSOR_TENSOR_H
