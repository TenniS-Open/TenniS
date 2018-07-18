//
// Created by seeta on 2018/5/19.
//

#ifndef TENSORSTACK_STACK_FUNCTION_H
#define TENSORSTACK_STACK_FUNCTION_H

#include <map>
#include <string>
#include <core/tensor.h>

namespace ts {
    class Stack;

    class Operator {
    public:
        using self = Operator;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        // explicit Operator(const std::string &name) : m_name(name) {}

        // work on tensor stack, same as call
        virtual int run(Stack &stack) = 0;

        // infer output size by input size, only for size allocate
        virtual int infer(Stack &stack, std::vector<Tensor::Prototype> &output) = 0;

        void set(const std::string &param, const Tensor &value);

        Tensor &get(const std::string &param);

        const Tensor &get(const std::string &param) const;

        void clear(const std::string &param);

        void clear_params();

    private:
        std::map<std::string, Tensor> m_params;
        // std::string m_name;
    };
}


#endif //TENSORSTACK_STACK_FUNCTION_H
