//
// Created by sen on 2021/9/15.
//
#include <runtime/runtime.h>
#include "sigmoid.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"
#include "hard_sigmoid.h"


namespace ts {
    namespace xnn {
        HardSigmoid::HardSigmoid() {
            field("alpha", OPTIONAL, tensor::from(float(0.2)));
            field("beta", OPTIONAL, tensor::from(float(0.5)));
        }

        void HardSigmoid::init() {
            supper::init();

            auto ctx = ctx::get<RuntimeContext>();
            m_threadpool = ctx->get_xnn_threadpool();

            m_alpha = tensor::to_float(get("alpha"));
            m_beta = tensor::to_float(get("beta"));
        }

        int HardSigmoid::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);
            auto cpu_hard_sigmoid_infer = OperatorCreator::Create(memory_device().type(), "hard_sigmoid", true);
            InferOperator(cpu_hard_sigmoid_infer, stack, 0, output);
            return 1;
        }

        int HardSigmoid::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);
            std::vector<Tensor::Prototype> output_protos;
//            infer(stack, output_protos);
            output_protos.resize(1);
            output_protos[0] = stack[0].proto();

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);

            auto out = *stack.push(output_protos[0], memory_device);

            hard_sigmoid(x, out);

            return 1;
        }

        void HardSigmoid::hard_sigmoid(const Tensor &x, Tensor &out) {
            float min = 0.f;
            float max = 1.f;
            if (m_mul_op == nullptr) {
                m_status = xnn_create_multiply_nd_f32(min, max, 0, &m_mul_op);
                TS_CHECK_EQ(m_status, xnn_status_success);
                m_shared_mul_op.reset(m_mul_op, xnn_delete_operator);
                m_mul_op = m_shared_mul_op.get();
            }
            if (m_add_op == nullptr) {
                m_status = xnn_create_add_nd_f32(min, max, 0, &m_add_op);
                TS_CHECK_EQ(m_status, xnn_status_success);
                m_shared_add_op.reset(m_add_op, xnn_delete_operator);
                m_add_op = m_shared_add_op.get();
            }

            std::vector<size_t> lhs_shape;
            std::vector<size_t> rhs_shape;
            lhs_shape.reserve(x.dims());
            rhs_shape.reserve(x.dims());
            for (int i = 0; i < x.dims(); ++i) { lhs_shape.emplace_back(1); }
            for (int i = 0; i < x.dims(); ++i) { rhs_shape.emplace_back(x.size(i)); }

            Tensor middle_tensor = x.clone();
            m_status = xnn_setup_multiply_nd_f32(m_mul_op, lhs_shape.size(), lhs_shape.data(), rhs_shape.size(),rhs_shape.data(),
                                                 &m_alpha, x.data<float>(), middle_tensor.data<float>(), m_threadpool);
            TS_CHECK(m_status == xnn_status_success);
            m_status = xnn_run_operator(m_mul_op, m_threadpool);
            TS_CHECK(m_status == xnn_status_success);

            m_status = xnn_setup_add_nd_f32(m_add_op, lhs_shape.size(), lhs_shape.data(), rhs_shape.size(),
                                            rhs_shape.data(), &m_beta, middle_tensor.data<float>(), out.data<float>(), m_threadpool);
            TS_CHECK(m_status == xnn_status_success);
            m_status = xnn_run_operator(m_add_op, m_threadpool);
            TS_CHECK(m_status == xnn_status_success);
        }
    }
}

using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(HardSigmoid, ts::XNNPACK, "xnn::hard_sigmoid")
