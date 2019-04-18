//
// Created by kier on 19-4-18.
//

#include "backend/garden/maskrcnn_anchor_generator.h"

#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"
#include "kernels/cpu/operator_on_cpu.h"

namespace ts {
    using namespace cpu;
    namespace maskrcnn {
        namespace cpu {
            class AnchorGenerator : public OperatorOnCPU<base::AnchorGenerator> {
            public:
                using self = AnchorGenerator;
                using supper = OperatorOnCPU<base::AnchorGenerator>;
            };
        }
    }
}

using namespace ts;
using namespace maskrcnn::cpu;

TS_REGISTER_OPERATOR(AnchorGenerator, CPU, AnchorGenerator::Name())
