#ifndef TENSORSTACK_KERNELS_OPT_OPERATOR_ON_OPT_H
#define TENSORSTACK_KERNELS_OPT_OPERATOR_ON_OPT_H

#include "backend/base/operator_on_device.h"

namespace ts {
    namespace opt {
        /**
         * @tparam OP must be the sub class of Operator or OperatorOnDevice
         */
        template<typename OP>
        class OperatorOnOPT : public OP {
        public:
            virtual MemoryDevice running_memory_device() {
                return MemoryDevice(CPU);
            }
        };
    }
}


#endif //TENSORSTACK_KERNELS_OPT_OPERATOR_ON_OPT_H
