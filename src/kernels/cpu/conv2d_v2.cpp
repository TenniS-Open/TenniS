#include <kernels/cpu/conv2d_v2.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Conv2DV2, CPU, name::layer::conv2d_v2())
