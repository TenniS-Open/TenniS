//
// Created by kier on 2019-04-13.
//

#include "frontend/compressor.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace desc {
        Bubble resize(ResizeType type) {
            Bubble bubble(name::layer::resize2d(), name::layer::resize2d());
            bubble.set(name::type, tensor::from(int32_t(type)));
            return std::move(bubble);
        }
    }
}
