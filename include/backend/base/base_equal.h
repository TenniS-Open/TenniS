#ifndef TENNIS_BASE_EQUAL_H
#define TENNIS_BASE_EQUAL_H

#include "element_wise_reduce.h"

namespace ts {
    namespace base {
        class Equal : public ElementWiseReduce {
        public:
            using self = Equal;
            using supper = ElementWiseReduce;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            int run(Stack &stack) override;
        };
    }
}

#endif //TENNIS_BASE_EQUAL_H
