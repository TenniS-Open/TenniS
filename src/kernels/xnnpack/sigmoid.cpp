//
// Created by sen on 2021/9/15.
//
#include <runtime/runtime.h>
#include "sigmoid.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"

namespace ts {
    namespace xnn {
        void Sigmoid::init() {
            supper::init();

            auto ctx = ctx::get<RuntimeContext>();
            m_threadpool = ctx->get_xnn_threadpool();
        }

        int Sigmoid::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);
            auto cpu_sigmoid_infer = OperatorCreator::Create(memory_device().type(), name::layer::sigmoid(), true);
            InferOperator(cpu_sigmoid_infer, stack, 0, output);
            return 1;
        }

        int Sigmoid::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);
            std::vector<Tensor::Prototype> output_protos;
//            infer(stack, output_protos);
            output_protos.resize(1);
            output_protos[0] = stack[0].proto();

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);

            auto out = *stack.push(output_protos[0], memory_device);

            sigmoid(x, out);

            return 1;
        }

        void Sigmoid::sigmoid(const Tensor &x, Tensor &out) {
            size_t channels = x.size(3);
            size_t batch_size = x.count() / channels;
            size_t input_stride = channels;
            size_t output_stride = channels;

            if (m_op == nullptr) {
                m_status = xnn_create_sigmoid_nc_f32(channels, input_stride, output_stride, 0, &m_op);
                TS_CHECK(m_status == xnn_status_success);
                m_shared_op.reset(m_op, xnn_delete_operator);
                m_op = m_shared_op.get();
            }

            m_status = xnn_setup_sigmoid_nc_f32(m_op, batch_size, x.data<float>(), out.data<float>(), m_threadpool);
            TS_CHECK(m_status == xnn_status_success);

            m_status = xnn_run_operator(m_op, m_threadpool);
            TS_CHECK(m_status == xnn_status_success);
        }
    }
}

using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(Sigmoid, ts::XNNPACK, "xnn::" + name::layer::sigmoid())
