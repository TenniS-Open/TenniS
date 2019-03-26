#include <kernels/gpu/conv2d.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(Conv2D, GPU, name::layer::conv2d())
