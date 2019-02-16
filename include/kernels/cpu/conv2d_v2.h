#ifndef TENSORSTACK_KERNELS_CPU_CONV2D_V2_H
#define TENSORSTACK_KERNELS_CPU_CONV2D_V2_H

#include "operator_on_cpu.h"
#include "backend/base/base_conv2d_v2.h"
#include "conv2d_core.h"


namespace ts {
    namespace cpu {
        using Conv2DV2 = base::Conv2DWithCore<OperatorOnCPU<base::Conv2DV2>, Conv2DCore>;
    }
}


#endif //TENSORSTACK_KERNELS_CPU_CONV2D_V2_H