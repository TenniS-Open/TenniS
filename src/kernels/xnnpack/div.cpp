//
// Created by sen on 2021/9/15.
//

#include <runtime/runtime.h>
#include "div.h"
#include "backend/name.h"
#include "global/operator_factory.h"

namespace ts {
    namespace xnn {
        void Div::init() {
            supper::init();

            auto ctx = ctx::get<ThreadPool>();
            m_threadpool = ctx->get_xnn_threadpool();

            if (m_op == nullptr) {
                float min = -std::numeric_limits<float>::infinity();
                float max = std::numeric_limits<float>::infinity();
                m_status = xnn_create_divide_nd_f32(min, max, 0, &m_op);
                TS_CHECK_EQ(m_status, xnn_status_success);
                m_shared_op.reset(m_op, xnn_delete_operator);
                m_op = m_shared_op.get();
            }
        }

        int Div::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            auto cpu_div_infer = OperatorCreator::Create(memory_device().type(), name::layer::div(), true);
            InferOperator(cpu_div_infer, stack, 2, output);
            return 1;
        }

        int Div::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            std::vector<Tensor::Prototype> output;
            auto lhs = *stack.index(0);
            auto rhs = *stack.index(1);

            infer(stack, output);

            auto memory_device = running_memory_device();

            lhs = lhs.view(memory_device);    // do sync, and set default data to given device
            rhs = rhs.view(memory_device);
            auto out = *stack.push(output[0], memory_device);

            div(lhs, rhs, out);

            return 1;
        }

        void Div::div(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            std::vector<size_t> lhs_shape;
            std::vector<size_t> rhs_shape;
            lhs_shape.reserve(lhs.dims());
            rhs_shape.reserve(rhs.dims());
            for (int i = 0; i < lhs.dims(); ++i) { lhs_shape.emplace_back(lhs.size(i)); }
            for (int i = 0; i < rhs.dims(); ++i) { rhs_shape.emplace_back(rhs.size(i)); }

            m_status = xnn_setup_divide_nd_f32(m_op, lhs_shape.size(), lhs_shape.data(), rhs_shape.size(),rhs_shape.data(),
                                                 lhs.data<float>(), rhs.data<float>(), out.data<float>(), m_threadpool);
            TS_CHECK(m_status == xnn_status_success);

            m_status = xnn_run_operator(m_op, m_threadpool);
            TS_CHECK(m_status == xnn_status_success);
        }


    }
}

using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(Div, ts::XNNPACK, "xnn::" + name::layer::div())
