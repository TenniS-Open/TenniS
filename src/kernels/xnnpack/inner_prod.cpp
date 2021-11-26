//
// Created by sen on 2021/11/17.
//

#include "inner_prod.h"
#include <runtime/runtime.h>
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"

namespace ts {
    namespace xnn {
        InnerProd::InnerProd() {
            field(name::alpha, OPTIONAL, tensor::from<float>(1.0f));
            field(name::beta, OPTIONAL, tensor::from<float>(1.0f));
            field(name::transA, OPTIONAL, tensor::from<bool>(false));
            field(name::transB, OPTIONAL, tensor::from<bool>(false));
        }

        void InnerProd::init() {
            supper::init();

            auto ctx = ctx::get<RuntimeContext>();
            m_threadpool = ctx->get_xnn_threadpool();
        }

        int InnerProd::infer(ts::Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);
            auto cpu_gemm_infer = OperatorCreator::Create(memory_device().type(), "gemm", true);
            InferOperator(cpu_gemm_infer, stack, 2, output);
            return 1;
        }

        int InnerProd::run(ts::Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 3);

            auto memory_device = running_memory_device();

            auto A = stack[0].view(memory_device);
            auto B = stack[1].view(memory_device);
            auto C = stack[2].view(memory_device);

            Tensor::Prototype output_proto = Tensor::Prototype(A.dtype(), {A.size(0), B.size(0)});;

            auto &out = *stack.push(output_proto, memory_device);

            inner_prod(A, B, C, out);

            return 1;
        }

        void InnerProd::inner_prod(const Tensor &A, const Tensor &B, const Tensor &C, Tensor &out) {
            size_t in_channels = A.size(1);
            size_t out_channels = out.size(1);
            size_t batch_size = A.count() / in_channels;
            size_t in_stride = in_channels;
            size_t out_stride = in_stride;

            if (m_op == nullptr) {
                float min = -std::numeric_limits<float>::infinity();
                float max = std::numeric_limits<float>::infinity();
                m_status = xnn_create_fully_connected_nc_f32(in_channels, out_channels, in_stride, out_stride,
                                                             B.data<float>(), C.data<float>(), min, max, 0, &m_op);
                TS_CHECK(m_status == xnn_status_success);
            }

            m_status = xnn_setup_fully_connected_nc_f32(m_op, batch_size, A.data<float>(), out.data<float>(), m_threadpool);
            TS_CHECK(m_status == xnn_status_success);

            m_status = xnn_run_operator(m_op, m_threadpool);
            TS_CHECK(m_status == xnn_status_success);
        }
    }
}
using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(InnerProd, ts::XNNPACK, "xnn::inner_prod")
