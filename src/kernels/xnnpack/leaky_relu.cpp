//
// Created by sen on 2021/9/15.
//
#include <runtime/runtime.h>
#include "leaky_relu.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"
#include "prelu.h"


namespace ts {
    namespace xnn {
        LeakyReLU::LeakyReLU() {
            field(name::scale, OPTIONAL, tensor::from(float(0)));
        }

        void LeakyReLU::init() {
            supper::init();

            auto ctx = ctx::get<RuntimeContext>();
            m_threadpool = ctx->get_xnn_threadpool();

            m_scale = tensor::to_float(get(name::scale));
        }

        int LeakyReLU::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto out = *stack.push(x.proto(), memory_device);

            leaky_relu(x, out);

            return 1;
        }

        int LeakyReLU::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto cpu_leaky_relu_infer = OperatorCreator::Create(memory_device().type(), "leaky_relu", true);
            InferOperator(cpu_leaky_relu_infer, stack, 1, output);

            return 1;
        }

        void LeakyReLU::leaky_relu(const Tensor &x, Tensor &out) {
            size_t channels = x.size(3);
            size_t batch_size = x.count() / channels;
            size_t input_stride = channels;
            size_t output_stride = channels;

            if (m_op == nullptr) {
                m_status = xnn_create_leaky_relu_nc_f32(channels, input_stride, output_stride, m_scale, 0, &m_op);
                TS_CHECK(m_status == xnn_status_success);
                m_shared_op.reset(m_op, xnn_delete_operator);
                m_op = m_shared_op.get();
            }

            m_status = xnn_setup_leaky_relu_nc_f32(m_op, batch_size, x.data<float>(), out.data<float>(), m_threadpool);
            TS_CHECK(m_status == xnn_status_success);

            m_status = xnn_run_operator(m_op, m_threadpool);
            TS_CHECK(m_status == xnn_status_success);
        }
    }
}

using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(LeakyReLU, ts::XNNPACK, "xnn::leaky_relu")