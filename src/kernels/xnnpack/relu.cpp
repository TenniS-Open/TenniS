//
// Created by sen on 2021/9/15.
//
#include <runtime/runtime.h>
#include "relu.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"
#include <tuple>

namespace ts {
    namespace xnn {

        void ReLU::init() {
            supper::init();

            auto ctx = ctx::get<RuntimeContext>();
            m_threadpool = ctx->get_xnn_threadpool();
        }

        int ReLU::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);
            auto cpu_relu_infer = OperatorCreator::Create(memory_device().type(), name::layer::relu(), true);
            InferOperator(cpu_relu_infer, stack, 0, output);
            return 1;
        }

        int ReLU::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);
            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto out = *stack.push(x.proto(), memory_device);

            relu(x, out);

            return 1;
        }


        void ReLU::relu(const Tensor &x, Tensor &out) {

            size_t channels = x.size(3);
            size_t batch_size = x.count() / channels;
            size_t input_stride = channels;
            size_t output_stride = channels;

            if (m_op == nullptr) {
                float min = 0.f;
                float max = std::numeric_limits<float>::infinity();
                m_status = xnn_create_clamp_nc_f32(channels, input_stride, output_stride, min, max, 0, &m_op);
                TS_CHECK(m_status == xnn_status_success);
                m_shared_op.reset(m_op, xnn_delete_operator);
            }

            m_status = xnn_setup_clamp_nc_f32(m_op, batch_size, x.data<float>(), out.data<float>(), m_threadpool);
            TS_CHECK(m_status == xnn_status_success);

            m_status = xnn_run_operator(m_op, m_threadpool);
            TS_CHECK(m_status == xnn_status_success);
        }
    }
}

using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(ReLU, ts::XNNPACK, "xnn::" + name::layer::relu())