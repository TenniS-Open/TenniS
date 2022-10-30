//
// Created by yang on 2019/11/12.
//

#include <core/tensor_builder.h>
#include "rknn2.h"

#include "global/operator_factory.h"
#include "backend/name.h"

namespace ts {
    namespace rknn {
        static void rknn2_forward_run(Rknn2 *self,
                                      const std::vector<Tensor> &inputs,
                                      const rknn_context &ctx,
                                      const int output_num,
                                      const DataFormat data_format,
                                      const std::vector<rknn_tensor_attr> &input_attrs,
                                      const std::vector<rknn_tensor_attr> &out_attrs,
                                      std::vector<Tensor> &outputs) {
            auto m_rknn = self->rknn();
            //build rknn inputs
            int input_num = inputs.size();
            std::unique_ptr<rknn_input[]> rknn_inputs(new rknn_input[input_num]);
            ::memset(rknn_inputs.get(), 0, sizeof(rknn_input) * input_num);
            for (int i = 0; i < input_num; ++i) {
                rknn_inputs[i].index = input_attrs[i].index;
                rknn_inputs[i].buf = const_cast<void *>(inputs[i].data());
                rknn_inputs[i].size = inputs[i].count() * type_bytes(inputs[i].dtype());
                rknn_inputs[i].type = get_rknn_type(inputs[i].dtype());
                rknn_inputs[i].fmt = input_attrs[i].fmt;    // format already checked in base RKNN2
                rknn_inputs[i].pass_through = 0;
            }
            int ret = m_rknn->inputs_set(ctx, input_num, rknn_inputs.get());
            if (ret != 0) {
                TS_LOG_ERROR << self->op() << " [RKNN] rknn_inputs_set failed!"
                             << " Error(" << ret << "): " << get_rknn_error_str(ret) << "." << eject;
            }

            //run on rknn
            ret = m_rknn->run(ctx, nullptr);
            if (ret != 0) {
                TS_LOG_ERROR << self->op() << " [RKNN] rknn_run failed!"
                             << " Error(" << ret << "): " << get_rknn_error_str(ret) << "." << eject;
            }

            //get rknn outputs
            std::unique_ptr<rknn_output[]> rknn_outputs(new rknn_output[output_num]);
            ::memset(rknn_outputs.get(), 0, sizeof(rknn_output) * output_num);
            for (int i = 0; i < output_num; ++i) {
                rknn_outputs[i].want_float = true;
                rknn_outputs[i].is_prealloc = true;
                rknn_outputs[i].index = i;
                rknn_outputs[i].buf = outputs[i].data();
                rknn_outputs[i].size = out_attrs[i].n_elems * sizeof(float);
            }

            ret = m_rknn->outputs_get(ctx, output_num, rknn_outputs.get(), nullptr);
            if (ret != 0) {
                TS_LOG_ERROR << self->op() << " [RKNN] rknn_outputs_get failed!"
                             << " Error(" << ret << "): " << get_rknn_error_str(ret) << "." << eject;
            }

            // for (int i = 0; i < output_num; ++i) {
            //     ::memset(outputs[i].data(), 0,
            //              outputs[i].count() * type_bytes(outputs[i].dtype()));
            // }

            m_rknn->outputs_release(ctx, output_num, rknn_outputs.get());
        }

        void Rknn2::rknn_forward(const std::vector<Tensor> &inputs,
                                 const rknn_context &ctx,
                                 const int output_num,
                                 const DataFormat data_format,
                                 const std::vector<rknn_tensor_attr> &input_attrs,
                                 const std::vector<rknn_tensor_attr> &out_attrs,
                                 std::vector<Tensor> &outputs) {
            // DTYPE dtype = inputs[0].dtype();
            rknn2_forward_run(this, inputs, ctx, output_num, data_format, input_attrs, out_attrs, outputs);
        }

    } //rknn
} //ts

using namespace ts;
using namespace rknn;
TS_REGISTER_OPERATOR(Rknn2, "rknn", "rknn2")

