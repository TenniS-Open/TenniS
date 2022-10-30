//
// Created by yang on 2019/11/12.
//

#ifndef TENSORSTACK_KERNELS_RKNN_RKNN2_H
#define TENSORSTACK_KERNELS_RKNN_RKNN2_H

#include "base_rknn2.h"
#include "operator_on_rknn.h"

namespace ts {
    namespace rknn {
        class Rknn2 : public OperatorOnRKNN<ts::base::Rknn2> {
        public:
            void rknn_forward(const std::vector<Tensor> &inputs,
                              const rknn_context &ctx,
                              const int output_num,
                              const DataFormat data_format,
                              const std::vector<rknn_tensor_attr> &input_attrs,
                              const std::vector<rknn_tensor_attr> &out_attrs,
                              std::vector<Tensor> &outputs);

        };
    }
}

#endif //TENSORSTACK_KERNELS_RKNN_RKNN2_H
