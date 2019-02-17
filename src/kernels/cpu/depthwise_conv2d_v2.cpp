#include <kernels/cpu/depthwise_conv2d_v2.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(DepthwiseConv2DV2, CPU, name::layer::depthwise_conv2d_v2())
