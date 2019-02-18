//
// Created by kier on 2019/2/18.
//

#include "backend/base/base_flatten.h"

#include <numeric>

namespace ts {
    namespace base {
        Shape Flatten::newshape(const Tensor &x) {
            TS_AUTO_CHECK(x.dims() > 0);

            auto &size = x.sizes();

            auto batch = x.size(0);
            auto number = std::accumulate(size.begin() + 1, size.end(), 1, std::multiplies<int>());

            return { batch, number };
        }
    }
}
