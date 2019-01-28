//
// Created by kier on 2019/1/28.
//

#ifndef TENSORSTACK_BACKEND_BASE_ELEMENT_WISE_REDUCE_H
#define TENSORSTACK_BACKEND_BASE_ELEMENT_WISE_REDUCE_H


#include <runtime/operator.h>

namespace ts {
    class ElementWiseReduce : public Operator {
    public:
        using self = ElementWiseReduce;
        using supper = Operator;

        ElementWiseReduce(const MemoryDevice &device);  // tell me the operator memory

        void init() override;

        int run(Stack &stack) override;

        int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        virtual void reduce_with_broadcast(const Tensor &lhs, const Tensor &rhs, Tensor &out) = 0;

        virtual void reduce_with_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out);

        /**
         *
         * @param lhs not specific
         * @param rhs is bias on dim
         * @param out
         */
        virtual void reduce_with_bias(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim);

        /**
         *
         * @param lhs not specific
         * @param rhs is scalar
         * @param out
         */
        virtual void reduce_with_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out);

        /**
         *
         * @param [in/out] lhs_shape lhs' shape
         * @param [in/out] rhs_shape rhs' shape
         * @param [out] out_shape output shape
         * @return if broadcast done
         */
        static bool reduce(Shape &lhs_shape, Shape &rhs_shape, Shape &out_shape, bool broadcast = true);

        static bool is_scalar(const Shape &shape);

        static bool is_bias(Shape &lhs_shape, Shape &rhs_shape, int &dim);

        const MemoryDevice &running_memory_device() const { return m_memory_device; }

    private:
        MemoryDevice m_memory_device;
    };
}


#endif //TENSORSTACK_BACKEND_BASE_ELEMENT_WISE_REDUCE_H
