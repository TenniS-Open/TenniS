//
// Created by sen on 2021/9/15.
//

#include <runtime/runtime.h>
#include "kernels/xnnpack/xnnpack.h"
#include "backend/base/operator_on_device.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"
#include "backend/common_function.h"
#include "runtime/operator.h"
#include "sub.h"

namespace ts {
    namespace xnn {
        void Sub::init() {
            supper::init();

            auto ctx = ctx::get<RuntimeContext>();
            m_threadpool = ctx->get_xnn_threadpool();

            if (m_op == nullptr) {
                float min = -std::numeric_limits<float>::infinity();
                float max = std::numeric_limits<float>::infinity();
                m_status = xnn_create_subtract_nd_f32(min, max, 0, &m_op);
                TS_CHECK_EQ(m_status, xnn_status_success);
                m_shared_op.reset(m_op, xnn_delete_operator);
            }
            m_op = m_shared_op.get();
        }

        int Sub::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            auto cpu_sub_infer = OperatorCreator::Create(memory_device().type(), name::layer::sub(), true);
            InferOperator(cpu_sub_infer, stack, 2, output);
            return 1;
        }

        int Sub::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            std::vector<Tensor::Prototype> output;
            auto lhs = *stack.index(0);
            auto rhs = *stack.index(1);

            infer(stack, output);

            auto memory_device = running_memory_device();

            lhs = lhs.view(memory_device);
            rhs = rhs.view(memory_device);
            auto out = *stack.push(output[0], memory_device);

            sub(lhs, rhs, out);

            return 1;
        }

        void Sub::sub(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            std::vector<size_t> lhs_shape;
            std::vector<size_t> rhs_shape;
            lhs_shape.reserve(lhs.dims());
            rhs_shape.reserve(rhs.dims());
            for (int i = 0; i < lhs.dims(); ++i) { lhs_shape.emplace_back(lhs.size(i)); }
            for (int i = 0; i < rhs.dims(); ++i) { rhs_shape.emplace_back(rhs.size(i)); }

            m_status = xnn_setup_subtract_nd_f32(m_op, lhs_shape.size(), lhs_shape.data(), rhs_shape.size(),
                                                 rhs_shape.data(), lhs.data<float>(), rhs.data<float>(), out.data<float>(), m_threadpool);
            TS_CHECK(m_status == xnn_status_success);

            m_status = xnn_run_operator(m_op, m_threadpool);
            TS_CHECK(m_status == xnn_status_success);
        }

    }
}

using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(Sub, ts::XNNPACK, "xnn::sub")
