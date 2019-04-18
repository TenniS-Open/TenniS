//
// Created by kier on 19-4-18.
//

#ifndef TENSORSTACK_BACKEND_GARDEN_MASKRCNN_ANCHOR_GENERATOR_H
#define TENSORSTACK_BACKEND_GARDEN_MASKRCNN_ANCHOR_GENERATOR_H


#include "backend/base/operator_on_device.h"

namespace ts {
    namespace maskrcnn {
        namespace base {
            class AnchorGenerator : public OperatorOnDevice {
            public:
                using self = AnchorGenerator;
                using supper = OperatorOnDevice;

                static std::string Name();

                void init() override;

                int infer(Stack &stack, std::vector<Tensor::Prototype> &outputs) override;

                int run(Stack &stack) override;
            };
        }
    }
}


#endif //TENSORSTACK_BACKEND_GARDEN_MASKRCNN_ANCHOR_GENERATOR_H
