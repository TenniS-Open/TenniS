//
// Created by kier on 2019/2/15.
//

#ifndef TENSORSTACK_BACKEND_BASE_OPERATOR_ON_DEVICE_H
#define TENSORSTACK_BACKEND_BASE_OPERATOR_ON_DEVICE_H

#include <runtime/operator.h>

namespace ts {
    class OperatorOnDevice : public Operator {
    public:
        virtual MemoryDevice running_memory_device() = 0;
    };
}


#endif //TENSORSTACK_BACKEND_BASE_OPERATOR_ON_DEVICE_H
