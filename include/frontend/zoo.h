//
// Created by kier on 2019/3/27.
//

#ifndef TENSORSTACK_FRONTEND_ZOO_H
#define TENSORSTACK_FRONTEND_ZOO_H

#include <module/graph.h>

namespace ts {
    namespace zoo {
        TS_DEBUG_API Node pad(const std::string &name, Node x, Node padding, float padding_value=0);
    }
}


#endif //TENSORSTACK_FRONTEND_ZOO_H
