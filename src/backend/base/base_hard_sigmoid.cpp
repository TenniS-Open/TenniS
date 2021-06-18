#include "backend/base/base_hard_sigmoid.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        HardSigmoid::HardSigmoid() {
            field("alpha", OPTIONAL, tensor::from(float(0.2)));
            field("beta", OPTIONAL, tensor::from(float(0.5)));
        }

        void HardSigmoid::init() {
            supper::init();

            m_alpha = tensor::to_float(get("alpha"));
            m_beta = tensor::to_float(get("beta"));
        }

        void HardSigmoid::active(const Tensor &x, Tensor &out) {
            hard_sigmoid(x, m_alpha, m_beta, out);
        }
    }
}

