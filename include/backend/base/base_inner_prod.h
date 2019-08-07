//
// Created by kier on 2019/2/18.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_INNER_PROD_H
#define TENSORSTACK_BACKEND_BASE_BASE_INNER_PROD_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class InnerProd : public OperatorOnDevice {
        public:
            using self = InnerProd;
            using supper = OperatorOnDevice;

            InnerProd();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param lhs
             * @param rhs
             * @param out
             * @note all tensor's dtype is same, and all tensors' memory device are give in constructor
             */
            virtual void inner_prod(const Tensor &lhs, const Tensor &rhs, bool transpose, Tensor &out, Stack &stack, bool kernel_need_pack) = 0;

        private:
            bool m_transpose = false;
            bool m_kernel_need_pack = true;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_INNER_PROD_H
