//
// Created by kier on 2019/3/6.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_GATHER_ELEMENTS_H
#define TENSORSTACK_BACKEND_BASE_BASE_GATHER_ELEMENTS_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class GatherElements : public OperatorOnDevice {
        public:
            using self = GatherElements;
            using supper = OperatorOnDevice;

            GatherElements();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param x input tensor
             * @param indices int32 tensor on CPU
             * @param axis action axis, in [0, x.dims())
             * @param out
             */
            virtual void gather(const Tensor &x, const Tensor &indices, int axis, Tensor &out) = 0;

        private:
            int m_axis = -1;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_GATHER_ELEMENTS_H
