//
// Created by kier on 19-4-18.
//

#include "backend/garden/maskrcnn_rpn_post_processor.h"

#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"
#include "kernels/cpu/operator_on_cpu.h"

namespace ts {
    using namespace cpu;
    namespace maskrcnn {
        namespace cpu {
            class RPNPostProcessor : public OperatorOnCPU<base::RPNPostProcessor> {
            public:
                using self = RPNPostProcessor;
                using supper =  OperatorOnCPU<base::RPNPostProcessor>;
            };
        }
    }
}

using namespace ts;
using namespace maskrcnn::cpu;

TS_REGISTER_OPERATOR(RPNPostProcessor, CPU, RPNPostProcessor::Name())


