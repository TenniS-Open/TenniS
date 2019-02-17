#ifndef TENSORSTACK_KERNELS_CPU_DEPTHWISE_CONV2D_V2_H
#define TENSORSTACK_KERNELS_CPU_DEPTHWISE_CONV2D_V2_H

#include "operator_on_cpu.h"
#include "backend/base/base_depthwise_conv2d_v2.h"
#include "depthwise_conv2d_core.h"


namespace ts {
    namespace cpu {
        using DepthwiseConv2DV2 = base::DepthwiseConv2DWithCore<OperatorOnCPU<base::DepthwiseConv2DV2>, DepthwiseConv2DCore>;
    }
}


#endif //TENSORSTACK_KERNELS_CPU_DEPTHWISE_CONV2D_V2_H