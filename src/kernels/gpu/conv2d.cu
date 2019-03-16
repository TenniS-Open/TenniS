#include <kernels/gpu/conv2d.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace gpu;
#ifdef TS_USE_CUBLAS
TS_REGISTER_OPERATOR(Conv2D, CUBLAS, name::layer::conv2d())
#else
TS_REGISTER_OPERATOR(Conv2D, GPU, name::layer::conv2d())
#endif
