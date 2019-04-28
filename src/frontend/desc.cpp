//
// Created by kier on 2019-04-13.
//

#include "frontend/desc.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace desc {
        Bubble resize2d(ResizeType type) {
            Bubble bubble(name::layer::resize2d(), name::layer::resize2d());
            bubble.set(name::type, tensor::from(int32_t(type)));
            return std::move(bubble);
        }

        Bubble add() {
            return Bubble(name::layer::add(), name::layer::add());
        }

        Bubble sub() {
            return Bubble(name::layer::sub(), name::layer::sub());
        }

        Bubble mul() {
            return Bubble(name::layer::mul(), name::layer::mul());
        }

        Bubble div() {
            return Bubble(name::layer::div(), name::layer::div());
        }
    }
}
