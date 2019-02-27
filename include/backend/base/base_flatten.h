//
// Created by kier on 2019/2/18.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_FLATTEN_H
#define TENSORSTACK_BACKEND_BASE_BASE_FLATTEN_H

#include "base_new_shape.h"

namespace ts {
    namespace base {
        class Flatten : public NewShape {
        public:
            Shape newshape(const Tensor &x) override;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_FLATTEN_H
