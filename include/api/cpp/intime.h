//
// Created by kier on 19-5-13.
//

#ifndef TENSORSTACK_API_CPP_INTIME_H
#define TENSORSTACK_API_CPP_INTIME_H

#include "tensor.h"
#include "../intime.h"

namespace ts {
    namespace api {
        namespace intime {
            inline Tensor transpose(const ts_Tensor *x, const std::vector<int32_t> &permute) {
                auto y = ts_intime_transpose(x, permute.data(), int32_t(permute.size()));
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor transpose(const Tensor &x, const std::vector<int32_t> &permute) {
                return transpose(x.get_raw(), permute);
            }

        }
    }
}

#endif //TENSORSTACK_API_CPP_INTIME_H
