//
// Created by kier on 2019/1/25.
//

#ifndef TENSORSTACK_BACKEND_HWC_CENTER_CROP2D_H
#define TENSORSTACK_BACKEND_HWC_CENTER_CROP2D_H


#include <runtime/operator.h>
#include "backend/common_structure.h"

namespace ts {
    namespace zoo {
        class HWCCenterCrop2D : public Operator {
        public:
            using self = HWCCenterCrop2D;
            using supper = Operator;

            HWCCenterCrop2D();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Size2D m_size;
            Operator::shared m_pad_op;
        };
    }
}


#endif //TENSORSTACK_BACKEND_HWC_CENTER_CROP2D_H
