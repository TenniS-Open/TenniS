//
// Created by seeta on 2018/5/25.
//

#ifndef TENSORSTACK_STACK_WORKBENCH_H
#define TENSORSTACK_STACK_WORKBENCH_H

#include <memory>

namespace ts {
    class workbench {
    public:
        using self = workbench;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer
    };
}


#endif //TENSORSTACK_STACK_WORKBENCH_H
