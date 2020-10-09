//
// Created by kier on 2019/1/25.
//

#include <runtime/operator.h>
#include "backend/common_structure.h"

#include "backend/name.h"

#include "core/tensor_builder.h"
#include "utils/assert.h"
#include "global/operator_factory.h"
#include "utils/ctxmgr_lite.h"
#include "core/device_context.h"
#include "runtime/stack.h"

namespace ts {
    namespace cpu {
        class CropTo : public Operator {
        public:
            using self = CropTo;
            using supper = Operator;

            CropTo();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Operator::shared m_pad_op;
            int m_axis;
            std::vector<int> m_offset;
        };

        CropTo::CropTo() {
            field("axis", OPTIONAL, tensor::build(INT32, 2));
            field("offset", OPTIONAL);
        }

        void CropTo::init() {
            supper::init();

            m_offset.clear();

            m_axis = tensor::to_int(get("axis"));
            if (has("offset")) {
                m_offset = tensor::array::to_int(get("offset"));
            }

            auto &context = ctx::ref<DeviceContext>();

            m_pad_op = OperatorCreator::Create(context.computing_device.type(), name::layer::pad(), false);

            TS_CHECK_NQ(m_pad_op, nullptr) << "Can not find operator: " << name::layer::pad();

            m_pad_op->set(name::padding_value, tensor::build(FLOAT32, {0.0f}));

            m_pad_op->init();
        }

        int CropTo::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x = *stack.index(0);
            auto &y = *stack.index(1);

            TS_AUTO_CHECK(x.dims() == y.dims());

            auto axis = m_axis >= 0 ? m_axis : m_axis += x.dims();

            if (axis < 0 || axis >= x.dims()) {
                TS_LOG_ERROR << "Crop axis must in [-"
                             << x.dims() << ", "
                             << x.dims() << ")" << eject;
            }

            auto shape = x.sizes();
            for (int i = axis; i < x.dims(); ++i) {
                shape[i] = y.size(i);
            }

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), shape);

            return 1;
        }

        int CropTo::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto x = *stack.index(0);
            auto y = *stack.index(1);

            // compute output shape

            TS_AUTO_CHECK(x.dims() == y.dims());

            auto axis = m_axis >= 0 ? m_axis : m_axis += x.dims();

            if (axis < 0 || axis >= x.dims()) {
                TS_LOG_ERROR << "Stack axis must in [-"
                             << x.dims() << ", "
                             << x.dims() << ")" << eject;
            }

            std::vector<int> offset = m_offset;
            if (!m_offset.empty()) {
                TS_AUTO_CHECK(offset.size() == x.dims() - axis);
            } else {
                offset.resize(x.dims() - axis, 0);
            }

            auto shape = x.sizes();
            for (int i = axis; i < x.dims(); ++i) {
                shape[i] = y.size(i);
            }

            // compute need padded size
            auto padding = std::vector<int>(axis * 2, 0);

            int offset_i = 0;
            for (int i = axis; i < x.dims(); ++i) {
                auto pad_left = -offset[offset_i];
                auto pad_right = y.size(i) - x.size(i) - pad_left;
                padding.push_back(pad_left);
                padding.push_back(pad_right);
                ++offset_i;
            }

            auto x_padding_tensor = tensor::build(INT32, {int(x.dims()), 2}, padding);

            stack.push(0);
            stack.push(x_padding_tensor);

            TS_AUTO_CHECK(1 == RunOperator(m_pad_op, stack, 2));

            return 1;
        }
    }
}

using namespace ts;
using namespace cpu;

TS_REGISTER_OPERATOR(CropTo, ts::CPU, "crop_to");
