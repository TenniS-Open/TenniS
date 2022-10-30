//
// Created by yang on 2019/11/13.
//

#ifndef TENSORSTACK_KERNELS_CPU_OPERATOR_ON_RKNN_H
#define TENSORSTACK_KERNELS_CPU_OPERATOR_ON_RKNN_H

#include "backend/base/operator_on_device.h"

namespace ts {
    namespace rknn {
        /**
         * @tparam OP must be the sub class of Operator or OperatorOnDevice
         */
        template<typename OP>
        class OperatorOnRKNN : public OP {
        public:
            virtual MemoryDevice running_memory_device() {
                return MemoryDevice(CPU);
            }
        };
    }
}

#endif //TENSORSTACK_KERNELS_CPU_OPERATOR_ON_RKNN_H
