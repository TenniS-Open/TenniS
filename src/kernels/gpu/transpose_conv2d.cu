#include <kernels/gpu/transpose_conv2d.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(Conv2DTranspose, GPU, name::layer::transpose_conv2d())
