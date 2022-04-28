#ifndef TENNIS_XNN_POOLING2D_H
#define TENNIS_XNN_POOLING2D_H

#include "kernels/cpu/operator_on_cpu.h"
#include <valarray>
#include "backend/common_structure.h"
#include "kernels/xnnpack/xnnpack.h"

namespace ts {
    namespace xnn {
        class Pooling2D : public cpu::OperatorOnCPU<OperatorOnDevice> {
        public:
            using self = Pooling2D;
            using supper = OperatorOnDevice;

            Pooling2D();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Conv2DFormat m_format;
            Pooling2DType m_type;
            std::valarray<int> m_padding4x2;
            Padding2DType m_padding_type;
            std::valarray<int> m_ksize4;
            std::valarray<int> m_stride4;
            
            pthreadpool_t m_threadpool = nullptr;
            xnn_operator_t m_op = nullptr;
            xnn_status m_status;
            std::shared_ptr<xnn_operator> m_shared_op;

            void pooling2d(const Tensor &x, Pooling2DType type,
                                   const Padding2D &padding, Padding2DType padding_type,
                                   const Size2D &ksize, const Stride2D &stride,
                                   Conv2DFormat format, Tensor &out);
        };
    }
}


#endif //TENNIS_XNN_POOLING2D_H
