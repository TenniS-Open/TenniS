//
// Created by kier on 2019-04-13.
//

#ifndef TENSORSTACK_FRONTEND_DESC_H
#define TENSORSTACK_FRONTEND_DESC_H

/**
 * compressor firstly convert params to bubble,
 * then frontend use it build graph
 *      intime use it run operator in time
 */

#include <module/bubble.h>

namespace ts {
    namespace desc {
        enum class ResizeType : int32_t {
            LINEAR = 0,
            CUBIC = 1,
            NEAREST = 2,
        };

        TS_DEBUG_API Bubble resize2d(ResizeType type = ResizeType::LINEAR);
    }
}

#endif //TENSORSTACK_FRONTEND_DESC_H
