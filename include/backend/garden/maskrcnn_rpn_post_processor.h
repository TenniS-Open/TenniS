//
// Created by kier on 19-4-18.
//

#ifndef TENSORSTACK_BACKEND_GARDEN_MASKRCNN_RPN_POST_PROCESSOR_H
#define TENSORSTACK_BACKEND_GARDEN_MASKRCNN_RPN_POST_PROCESSOR_H


#include "backend/base/operator_on_device.h"

namespace ts {
    namespace maskrcnn {
        namespace base {
            class RPNPostProcessor : public OperatorOnDevice {
            public:
                using self = RPNPostProcessor;
                using supper = OperatorOnDevice;

                static std::string Name();

                void init() override;

                int infer(Stack &stack, std::vector<Tensor::Prototype> &outputs) override;

                int run(Stack &stack) override;
            };
        }
    }
}


#endif //TENSORSTACK_BACKEND_GARDEN_MASKRCNN_RPN_POST_PROCESSOR_H
