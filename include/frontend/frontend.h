//
// Created by kier on 2019/3/27.
//

#ifndef TENSORSTACK_FRONTEND_FRONTEND_H
#define TENSORSTACK_FRONTEND_FRONTEND_H

#include "module/menu.h"

#include "symbol.h"

#include <module/graph.h>

#include "frontend/desc.h"

namespace ts {
    namespace symbol {
        using namespace bubble;
        struct DimPadding {
            DimPadding() = default;
            DimPadding(int32_t first, int32_t second)
                : first(first), second(second) {}

            int32_t first = 0;
            int32_t second = 0;
        };

        TS_DEBUG_API Node pad(const std::string &name, Node x, const std::vector<DimPadding> &padding, float padding_value=0);
    }
}


#endif //TENSORSTACK_FRONTEND_FRONTEND_H
