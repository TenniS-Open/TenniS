#ifndef TENNIS_GLOBAL_POOLING2D_H
#define TENNIS_GLOBAL_POOLING2D_H

#include "kernels/cpu/operator_on_cpu.h"
#include <valarray>
#include "backend/common_structure.h"
#include "kernels/xnnpack/xnnpack.h"

namespace ts {
    namespace xnn {
        class GlobalPooling2D : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = GlobalPooling2D;
            using supper = OperatorOnDevice;

            GlobalPooling2D();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Conv2DFormat m_format;
            Pooling2DType m_type;
            pthreadpool_t m_threadpool = nullptr;
            xnn_operator_t m_op = nullptr;
            xnn_status m_status;

            void pooling2d(const Tensor &x, Pooling2DType type, Padding2D padding, Padding2DType padding_type,
                           Size2D ksize, Stride2D stride, Conv2DFormat format, Tensor &out);
        };
    }
}


#endif //TENNIS_GLOBAL_POOLING2D_H
