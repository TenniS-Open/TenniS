//
// Created by yang on 2019/11/12.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_RKNN2_H
#define TENSORSTACK_BACKEND_BASE_BASE_RKNN2_H

#include "backend/base/operator_on_device.h"
#include "backend/common_structure.h"
#include "rknn_utils.h"

namespace ts {
    namespace base {
        class Rknn2 : public OperatorOnDevice {
        public:
            using self = Rknn2;
            using supper = OperatorOnDevice;

            Rknn2();

            ~Rknn2();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void rknn_forward(const std::vector<Tensor> &inputs,
                                      const rknn_context &ctx,
                                      const int output_num,
                                      const DataFormat data_format,
                                      const std::vector<rknn_tensor_attr> &input_attrs,
                                      const std::vector<rknn_tensor_attr> &out_attrs,
                                      std::vector<Tensor> &outputs) = 0;

        private:
            int m_input_num;
            int m_out_num;
            std::string m_model_path;
            DataFormat m_data_format;

            //rknn context
            rknn_context m_ctx = 0;
            //rknn attribute
            std::vector<rknn_tensor_attr> m_in_attrs;
            std::vector<rknn_tensor_attr> m_out_attrs;

            Operator::shared m_nchw2nhwc;
            Operator::shared m_nhwc2nchw;

            Operator::shared _get_op_nchw2nhwc();
            Operator::shared _get_op_nhwc2nchw();

        protected:
            RknnDll *m_rknn;

        public:
            RknnDll *rknn() { return m_rknn; }
        };
    }
}

#endif //TENSORSTACK_BACKEND_BASE_BASE_RKNN2_H
