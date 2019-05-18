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
            inline Tensor transpose(const Tensor &x, const std::vector<int32_t> &permute) {
                auto y = ts_intime_transpose(x.get_raw(), permute.data(), int32_t(permute.size()));
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor sigmoid(const Tensor &x) {
                auto y = ts_intime_sigmoid(x.get_raw());
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor gather(const Tensor &x, const Tensor &indices, int32_t axis) {
                auto y = ts_intime_gather(x.get_raw(), indices.get_raw(), axis);
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor concat(const std::vector<Tensor> &x, int32_t dim) {
                std::vector<ts_Tensor*> inputs;
                for (auto &input : x) {
                    inputs.emplace_back(input.get_raw());
                }
                auto y = ts_intime_concat(inputs.data(), int32_t(x.size()), dim);
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor softmax(const Tensor &x, int32_t dim, bool smooth = true) {
                auto y = ts_intime_softmax(x.get_raw(), dim, ts_bool(smooth));
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }
        }
    }
}

#endif //TENSORSTACK_API_CPP_INTIME_H
