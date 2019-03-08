#include <kernels/gpu/depthwise_conv2d.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(DepthwiseConv2D, GPU, name::layer::depthwise_conv2d())
