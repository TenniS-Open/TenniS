//
// Created by kier on 2018/10/31.
//

#ifndef TENSORSTACK_MODULE_MENU_H
#define TENSORSTACK_MODULE_MENU_H

#include "module.h"
#include "utils/ctxmgr.h"

namespace ts {
    namespace bubble {
        Node param(const std::string &name);
        Node op(const std::string &name, const std::string &op_name, const std::vector<Node> &inputs);
        Node data(const std::string &name, const Tensor &value);
    }
}


#endif //TENSORSTACK_MODULE_MENU_H
