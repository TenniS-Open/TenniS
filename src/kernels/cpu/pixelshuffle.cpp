//
// Created by sen on 2022/1/25.
//

#include "kernels/cpu/pixelshuffle.h"
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <core/device.h>

#include "module/bubble.h"
#include "runtime/workbench.h"

namespace ts {
    namespace cpu {
        void PixelShuffle::pixel_shuffle(const Tensor &x, Tensor &out, int scale_factor, std::string mode) {
            int in_h = x.size(2);
            int in_w = x.size(3);
            int out_ch = out.size(1);

            // CRD mode
            if (mode == "CRD") {
                auto tmp_tensor = x.reshape({x.size(0), out_ch, scale_factor, scale_factor, in_h, in_w});
                auto ctx_bench = ts::ctx::get<ts::Workbench>();
                std::vector<ts::Tensor> run_output;
                auto ps_bubble = ts::Bubble("_transpose", 1);
                ps_bubble.set("permute", ts::tensor::from({0, 1, 4, 2, 5, 3}));
                auto op = ctx_bench->offline_create(ps_bubble, false);
                ctx_bench->offline_run(op, {tmp_tensor}, run_output);
                out = run_output[0].reshape({x.size(0), out_ch, in_h * scale_factor, in_w * scale_factor});
            } else {
            // DCR mode
                auto tmp_tensor = x.reshape({x.size(0), scale_factor, scale_factor, out_ch, in_h, in_w});
                auto ctx_bench = ts::ctx::get<ts::Workbench>();
                std::vector<ts::Tensor> run_output;
                auto ps_bubble = ts::Bubble("_transpose", 1);
                ps_bubble.set("permute", ts::tensor::from({0, 3, 4, 1, 5, 2}));
                auto op = ctx_bench->offline_create(ps_bubble, false);
                ctx_bench->offline_run(op, {tmp_tensor}, run_output);
                out = run_output[0].reshape({x.size(0), out_ch, in_h * scale_factor, in_w * scale_factor});
            }

        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(PixelShuffle, ts::CPU, "pixel_shuffle")
