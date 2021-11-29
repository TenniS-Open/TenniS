//
// Created by sen on 2021/9/15.
//
#include <runtime/runtime.h>
#include "prelu.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"


namespace ts {
    namespace xnn {
        PReLU::PReLU() {
            field(name::dim, REQUIRED);
        }

        void PReLU::init() {
            // supper::init();
            // m_status = xnn_initialize(nullptr);
            // TS_CHECK(m_status == xnn_status_success);
            auto ctx = ctx::get<RuntimeContext>();
            m_threadpool = ctx->get_xnn_threadpool();

            m_dim = tensor::to_int(this->get(name::dim));
            TS_AUTO_CHECK(m_dim >= 0);
        }

        int PReLU::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto cpu_prelu_infer = OperatorCreator::Create(memory_device().type(), name::layer::prelu(), true);
            InferOperator(cpu_prelu_infer, stack, 1, output);

            return 1;
        }

        bool PReLU::check_inputs(Stack &stack) const {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x = stack[0];
            auto &slope = stack[1];

            TS_AUTO_CHECK(m_dim < x.dims());

            int check_dim = m_dim;
            switch (m_dim) {
                case 1: check_dim = 3; break;
                case 2: check_dim = 1; break;
                case 3: check_dim = 2; break;
                default: check_dim = m_dim; break;
            }

            if(!slope.has_shape(x.size(check_dim))) {
                TS_LOG_ERROR << "Miss matched: x:" << to_string(x.sizes()) <<
                             ", dim=" << check_dim <<
                             ", slope:" << to_string(slope.sizes()) << eject;
            }

            TS_AUTO_CHECK(x.dtype() == slope.dtype());
            return true;
        }

        int PReLU::run(Stack &stack) {
            check_inputs(stack);
            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto slope = stack[1].view(memory_device);
            auto out = *stack.push(x.proto(), memory_device);

            slope = tensor::cast(FLOAT32, slope);
            prelu(x, slope, out);

            return 1;
        }

        void PReLU::prelu(const Tensor &x, const Tensor &slope, Tensor &out) {
            size_t channels = x.size(3);
            size_t batch_size = x.count() / channels;
            size_t input_stride = channels;
            size_t output_stride = channels;

            if (m_op == nullptr) {
                m_status = xnn_create_prelu_nc_f32(channels, input_stride, output_stride, slope.data<float>(), 0, &m_op);
                TS_CHECK(m_status == xnn_status_success);
                m_shared_op.reset(m_op, xnn_delete_operator);
                m_op = m_shared_op.get();
            }

            m_status = xnn_setup_prelu_nc_f32(m_op, batch_size, x.data<float>(), out.data<float>(), m_threadpool);
            TS_CHECK(m_status == xnn_status_success);

            m_status = xnn_run_operator(m_op, m_threadpool);
            TS_CHECK(m_status == xnn_status_success);
        }
    }
}

using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(PReLU, ts::XNNPACK, "xnn::" + name::layer::prelu())