#ifndef TENNIS_BACKEND_BASE_BASE_POW_H
#define TENNIS_BACKEND_BASE_BASE_POW_H


#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Pow : public OperatorOnDevice {
        public:
            using self = Pow;
            using supper = OperatorOnDevice;

            Pow();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void pow(const Tensor &x, float y, Tensor &out) = 0;

        private:
            float m_y = 1;
        };
    }
}


#endif //TENNIS_BACKEND_BASE_BASE_POW_H
