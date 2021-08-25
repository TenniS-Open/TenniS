#ifndef TENNIS_BACKEND_BASE_BASE_HARD_SIGMOID_H
#define TENNIS_BACKEND_BASE_BASE_HARD_SIGMOID_H

#include "base_activation.h"

namespace ts {
    namespace base {
        class HardSigmoid : public Activation {
        public:
            HardSigmoid();

            void init() override;

            void active(const Tensor &x, Tensor &out) final;

            virtual void hard_sigmoid(const Tensor &x, float alpha, float beta, Tensor &out) = 0;

        private:
            float m_alpha = 0.2f;
            float m_beta = 0.5f;
        };
    }
}


#endif //TENNIS_BACKEND_BASE_BASE_HARD_SIGMOID_H
