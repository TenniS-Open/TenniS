//
// Created by kier on 2019/3/5.
//

#include "backend/base/base_unsqueeze.h"
#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        Unsqueeze::Unsqueeze() {
            field(name::axes, REQUIRED);
        }

        void Unsqueeze::init() {
            supper::init();

            auto axes_tensor = tensor::cast(INT32, get(name::axes));

            TS_AUTO_CHECK(axes_tensor.dims() == 1);

            size_t count = size_t(axes_tensor.size(0));

            m_axes.clear();
            m_axes.reserve(count);

            for (size_t i = 0; i < count; ++i) {
                m_axes.emplace_back(axes_tensor.data<int32_t>(i));
            }

            for (size_t i = 0; i < count; ++i) {
                if (m_axes[i] < 0) {
                    TS_LOG_ERROR << op() << " do not support unsqueeze shape "
                                 << "with axes=" << to_string(m_axes) << eject;
                }
            }

        }

        Shape Unsqueeze::newshape(const Tensor &x) {
            auto shape = x.sizes();

            for (auto axis : m_axes) {
                if (axis > shape.size()) {
                    TS_LOG_ERROR << op() << " do not support unsqueeze shape=" << to_string(x.sizes())
                                 << " with axes=" << to_string(m_axes) << eject;
                }
                shape.insert(shape.begin() + axis, 1);
            }

            return shape;
        }
    }
}
