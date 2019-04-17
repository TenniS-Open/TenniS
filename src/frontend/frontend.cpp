//
// Created by kier on 2019/3/27.
//

#include "frontend/frontend.h"

#include "module/menu.h"
#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace symbol {
        Node pad(const std::string &name, Node x, const std::vector<DimPadding> &padding, float padding_value) {
            TS_AUTO_CHECK(padding.size() > 0);
            Node padding_node = bubble::data("_const_" + name + "_padding",
                    tensor::build(INT32, {int(padding.size()), 2}, &padding[0].first), DeviceType(CPU));
            return pad(name, x, padding_node, padding_value);
        }
    }
}