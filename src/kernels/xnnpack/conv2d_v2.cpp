#include "conv2d_v2.h"
#include <global/operator_factory.h>
#include <backend/name.h>

#include "utils/ctxmgr_lite.h"
#include "core/device_context.h"

#include "module/bubble.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace xnn {

        Conv2DV2::Conv2DV2() {
            field(name::format, REQUIRED);
            field(name::padding_value, OPTIONAL, tensor::from(0.0f));
            field(name::stride, REQUIRED);
            field(name::dilation, OPTIONAL);
            field(name::typo::dialations, OPTIONAL);
            field(name::kernel_packed, OPTIONAL, tensor::from<bool>(false));
            field("bias", OPTIONAL, tensor::from(0.0f));
            field("groups", OPTIONAL, tensor::from(1));
            field("value_max", OPTIONAL, tensor::from(+std::numeric_limits<float>::infinity()));
            field("value_min", OPTIONAL, tensor::from(-std::numeric_limits<float>::infinity()));
        }

        void Conv2DV2::init() {
            supper::init();

            m_op_xnn_conv2d = OperatorCreator::Create(ts::CPU, "xnn::" + name::layer::conv2d(), true);

            TS_CHECK_NQ(m_op_xnn_conv2d, nullptr) << "Can not find operator: " << "xnn::" + name::layer::conv2d();

            m_op_xnn_conv2d->set(Bubble::RetentionParam::op, tensor::from("xnn::" + name::layer::conv2d_v2()));
            m_op_xnn_conv2d->set(Bubble::RetentionParam::name, tensor::from("_core" + name()));
            for (auto &param : Bubble::RetentionParam::All()) {
                if (!m_op_xnn_conv2d->has(param) && this->has(param)) {
                    m_op_xnn_conv2d->set(param, get(param));
                }
            }

            m_op_xnn_conv2d->set(name::format, get(name::format));
            m_op_xnn_conv2d->set(name::padding_value, get(name::padding_value));
            m_op_xnn_conv2d->set(name::stride, get(name::stride));
            m_op_xnn_conv2d->set(name::kernel_packed, get(name::kernel_packed));
            m_op_xnn_conv2d->set("bias", get("bias"));
            m_op_xnn_conv2d->set("groups", get("groups"));
            m_op_xnn_conv2d->set("value_max", get("value_max"));
            m_op_xnn_conv2d->set("value_min", get("value_min"));

            if (has(name::dilation)) m_op_xnn_conv2d->set(name::dilation, get(name::dilation));
            if (has(name::typo::dialations)) m_op_xnn_conv2d->set(name::typo::dialations, get(name::typo::dialations));
        }

        static bool is_int_equal(const Tensor &lhs, const Tensor &rhs) {
            if (!lhs.has_shape(rhs.sizes())) return false;
            auto count = lhs.count();
            for (int i = 0; i < count; ++i) {
                if(lhs.data<int>(i) != rhs.data<int>(i)) return false;
            }
            return true;
        }

        int Conv2DV2::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 3);

            auto padding = tensor::cast(INT32, stack[1]);

            if (!is_int_equal(padding, m_int_padding4x2)) {
                m_int_padding4x2 = padding.clone();
                m_op_xnn_conv2d->set(name::padding, m_int_padding4x2);
                m_op_xnn_conv2d->init();
            }

            stack.push(0);
            stack.push(2);

            return InferOperator(m_op_xnn_conv2d, stack, 2, output);
        }

        int Conv2DV2::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 3);

            auto padding = tensor::cast(INT32, stack[1]);

            if (!is_int_equal(padding, m_int_padding4x2)) {
                m_int_padding4x2 = padding.clone();
                m_op_xnn_conv2d->set(name::padding, m_int_padding4x2);
                m_op_xnn_conv2d->init();
            }

            stack.push(0);
            stack.push(2);

            return RunOperator(m_op_xnn_conv2d, stack, 2);
        }
    }
}

using namespace ts;
using namespace xnn;
TS_REGISTER_OPERATOR(Conv2DV2, CPU, "xnn::" + name::layer::conv2d_v2())
