#include <kernels/cpu/conv2d_winograd.h>

#include "kernels/cpu/conv2d_winograd.h"
#include "global/operator_factory.h"
#include "backend/name.h"
#include <kernels/cpu/conv2d_algorithm.h>

namespace ts {
    namespace cpu{
        template<typename T>
        static void conv2d_winograd_forward(const Tensor &x, WinogradConv2DMode winograd_model,
            const Tensor &w, Tensor &out, Stack &stack) {
            auto x_shape = x.sizes();
            auto weight_shape = w.sizes();
            Size2D ksize(weight_shape[2], weight_shape[3]);

            if (winograd_model == F6X6_3X3)
                return Conv2dAlgorithm<T>::conv3x3_winograd63(x, w, out);
            else
                return Conv2dAlgorithm<T>::conv3x3_winograd23(x, w, out);

            //if (padding.bottom != 0 || padding.top != 0 || padding.left != 0 || padding.right != 0) {
            //    auto pad_operator = OperatorCreator::Query(CPU, name::layer::pad())();
            //    TS_CHECK_NQ(pad_operator, nullptr) << "Can not find operator: " << name::layer::pad();

            //    ts::Shape pad_shape = { 4,2 };
            //    auto input_pad_param = tensor::build(INT32, pad_shape, { 0, 0, 0, 0, padding.top, padding.bottom, padding.left, padding.right });
            //    pad_operator->set(name::padding_value, tensor_builder<float>::build(&padding_value, 1));
            //    pad_operator->init();
            //    stack.push(x);
            //    stack.push(input_pad_param);
            //    RunOperator(pad_operator, stack, 2);
            //    auto x_pad_tensor = *(stack.top());
            //    stack.pop();

            //    if (winograd_mode == F6X6_3X3)
            //        return Conv2dAlgorithm<T>::conv3x3_winograd63(x_pad_tensor, w, out);
            //    else
            //        return Conv2dAlgorithm<T>::conv3x3_winograd23(x_pad_tensor, w, out);
            //}
            //else {
            //    if (winograd_mode == F6X6_3X3)
            //        return Conv2dAlgorithm<T>::conv3x3_winograd63(x, w, out);
            //    else
            //        return Conv2dAlgorithm<T>::conv3x3_winograd23(x, w, out);
            //}
        }

        void Conv2DWinograd::conv2d_winograd(const Tensor &x, WinogradConv2DMode winograd_mode,
            const Tensor &w, Conv2DFormat format, Tensor &out, Stack &stack) {
            if (format != FORMAT_NCHW) {
                TS_LOG_ERROR << "Conv2D_Winograd only support NCHW" << eject;
            }
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { conv2d_winograd_forward<TYPE>(x, winograd_mode, w, out, stack); break; }
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
            default: {
                TS_LOG_ERROR << "Conv2D not support this data type: " << type_str(dtype) << eject;
                break;
                }
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Conv2DWinograd, ts::CPU, name::layer::conv2d_winograd())