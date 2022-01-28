//
// Created by sen on 2022/1/25.
//

#include "pixelshuffle.h"
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <core/device.h>
#include "runtime/runtime.h"
#include "runtime/workbench.h"

namespace ts {
    namespace xnn {
        PixelShuffle::PixelShuffle() {
            field("upscale_factor", REQUIRED, tensor::from<int32_t>(1));
            field("mode", OPTIONAL, tensor::from("DCR"));
        }

        void PixelShuffle::init() {
            supper::init();

            m_mode = tensor::to_string(get("mode"));
            m_scale_factor = tensor::to_int(get("upscale_factor"));
            auto ctx = ctx::get<RuntimeContext>();
            m_threadpool = ctx->get_xnn_threadpool();
        }

        int PixelShuffle::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            auto x_shape = stack[0].sizes();
            TS_CHECK(x_shape.size() == 4);

            x_shape[3] /= m_scale_factor * m_scale_factor;
            x_shape[1] *= m_scale_factor;
            x_shape[2] *= m_scale_factor;

            output.emplace_back(TensorPrototype(stack[0].dtype(), x_shape));
            return 1;
        }

        void PixelShuffle::pixel_shuffle(const Tensor &x, Tensor &out, int scale_factor, std::string mode) {
            if (mode == "CRD") {
//                TS_LOG_ERROR << "PixelShuffle operator only support DCR mode on xnnpack backend." << eject;
                int in_h = x.size(1);
                int in_w = x.size(2);
                int out_ch = out.size(3);

                auto tmp_tensor = x.reshape({x.size(0), in_h, in_w, out_ch, scale_factor, scale_factor});
                auto ctx_bench = ts::ctx::get<ts::Workbench>();
                std::vector<ts::Tensor> run_output;
                auto ps_bubble = ts::Bubble("_transpose", 1);
                ps_bubble.set("permute", ts::tensor::from({0, 1, 4, 2, 5, 3}));
                auto op = ctx_bench->offline_create(ps_bubble, false);
                ctx_bench->offline_run(op, {tmp_tensor}, run_output);
                out = run_output[0].reshape({x.size(0), in_h * scale_factor, in_w * scale_factor, out_ch});
            } else {
                if (m_op == nullptr) {
                    size_t output_channels = out.size(3);
                    size_t input_channel_stride = x.size(3);
                    size_t output_channel_stride = out.size(3);
                    uint32_t block_size = scale_factor;
                    uint32_t flags = 0;
                    m_status = xnn_create_depth_to_space_nhwc_x32(output_channels, input_channel_stride, output_channel_stride, block_size, flags, &m_op);
                    TS_CHECK(m_status == xnn_status_success);
                    m_shared_op.reset(m_op, xnn_delete_operator);
                    m_op = m_shared_op.get();
                }

                size_t batch_size = x.size(0);
                size_t input_height = x.size(1);
                size_t input_width = x.size(2);
                m_status = xnn_setup_depth_to_space_nhwc_x32(m_op, batch_size, input_height, input_width, x.data(), out.data(), m_threadpool);
                TS_CHECK(m_status == xnn_status_success);

                m_status = xnn_run_operator(m_op, m_threadpool);
                TS_CHECK(m_status == xnn_status_success);
            }
        }
    }
}

using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(PixelShuffle, ts::XNNPACK, "xnn::pixel_shuffle")
