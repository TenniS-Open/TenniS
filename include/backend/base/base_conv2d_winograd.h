#ifndef TENSORSTACK_BACKEND_BASE_BASE_CONV2D_WINOGRAD_H
#define TENSORSTACK_BACKEND_BASE_BASE_CONV2D_WINOGRAD_H

#include "operator_on_device.h"
#include "backend/common_structure.h"

#include <valarray>

namespace ts{
    namespace base {
        class Conv2DWinograd : public OperatorOnDevice {
        public:
            using self = Conv2DWinograd;
            using supper = OperatorOnDevice;

            Conv2DWinograd();

            void init() override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            int run(ts::Stack &stack) override;

            virtual void conv2d_winograd(const Tensor &x, WinogradConv2DModel winograd_model, const Padding2D &padding, float padding_value,
                const Tensor &w, Conv2DFormat format, Tensor &out, Stack &stack) = 0;

        private:
            WinogradConv2DModel m_winograd_model;
            Conv2DFormat m_format;
            std::valarray<int> m_padding4x2;
            float m_padding_value;
        };
    }
}

#endif