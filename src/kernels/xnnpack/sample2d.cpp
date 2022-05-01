#include "core/dtype.h"
#include "kernels/xnnpack/xnnpack.h"
#include "sample2d.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"
#include "utils/assert.h"
#include "utils/log.h"
#include "backend/common_structure.h"
#include "core/device_context.h"
#include <cstdint>

namespace ts {
    namespace xnn {

        Sample2D::Sample2D() {
            field(name::scale, REQUIRED);
            field(name::type, OPTIONAL, tensor::from<int32_t>(2));  // NEAREST method
            field(name::dim, OPTIONAL, tensor::from<int32_t>(-2));
        }

        void Sample2D::init() {
            supper::init();

            m_scale = tensor::to_float(get(name::scale));
            m_dim = tensor::to_int(get(name::dim));

            if (m_scale < 1e-5) {
                TS_LOG_ERROR << "sample scale must greater than 1e-5, got" << m_scale << eject;
            }

            auto &context = ctx::ref<DeviceContext>();

            m_sample_op = OperatorCreator::Create(context.computing_device.type(), name::layer::affine_sample2d(), false);

            TS_CHECK_NQ(m_sample_op, nullptr) << "Can not find operator: " << name::layer::affine_sample2d();

            m_sample_op->set(name::type, get(name::type).clone());
            m_sample_op->set(name::outer_value, tensor::from<float>(0));
            auto affine_dim = get(name::dim).clone();
            // nhwc layout
            // TS_CHECK(m_dim == 2 || m_dim == -2);
            affine_dim.data<int>(0) = -3;
            m_sample_op->set(name::dim, affine_dim);

            m_sample_op->init();

            m_sample_size = Tensor(INT32, {2, });
            m_sample_affine = Tensor(FLOAT32, {3, 3});

            auto *affine = m_sample_affine.data<float>();
            for (int i = 0; i < 8; ++i) affine[i] = 0;
            affine[8] = 1;
            affine[0] = 1 / m_scale;
            affine[4] = 1 / m_scale;
        }

        int Sample2D::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];

            const auto max_dim = int(x.dims());
            auto fixed_dim = m_dim < 0 ? max_dim + m_dim : m_dim;

            if (fixed_dim < 0 || fixed_dim >= (max_dim - 1)) {
                TS_LOG_ERROR << "Sample2D dim must in [-"
                             << max_dim << ", "
                             << max_dim - 1 << ")" << eject;
            }

            auto shape = x.sizes();
            // nhwc layout
            fixed_dim -= 1;
            shape[fixed_dim] = int(shape[fixed_dim] * m_scale);
            shape[fixed_dim + 1] = int(shape[fixed_dim + 1] * m_scale);

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), shape);

            return 1;
        }

        int Sample2D::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];

            const auto max_dim = int(x.dims());
            auto fixed_dim = m_dim < 0 ? max_dim + m_dim : m_dim;

            if (fixed_dim < 0 || fixed_dim >= (max_dim - 1)) {
                TS_LOG_ERROR << "Sample2D dim must in [-"
                             << max_dim << ", "
                             << max_dim - 1 << ")" << eject;
            }

            auto &shape = x.sizes();

            // nhwc layout
            fixed_dim -= 1;

            m_sample_size.data<int32_t>(0) = int32_t(std::floor(shape[fixed_dim] * m_scale));
            m_sample_size.data<int32_t>(1) = int32_t(std::floor(shape[fixed_dim + 1] * m_scale));

            stack.push(m_sample_size);
            stack.push(m_sample_affine);

            TS_AUTO_CHECK(1 == RunOperator(m_sample_op, stack, 3));

            return 1;
        }
    }
}

using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(Sample2D, ts::XNNPACK, "xnn::sample2d")
