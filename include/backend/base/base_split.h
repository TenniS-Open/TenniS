#ifndef TENNIS_BACKEND_BASE_BASE_SPLIT_H
#define TENNIS_BACKEND_BASE_BASE_SPLIT_H


#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Split : public OperatorOnDevice {
        public:
            using self = Split;
            using supper = OperatorOnDevice;

            Split();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void split(const Tensor &x, const std::vector<int> &split, int dim,
                               std::vector<Tensor> &out) = 0;

        private:
            int m_dim = -1;
        };
    }
}


#endif //TENNIS_BACKEND_BASE_BASE_SPLIT_H
