#include <kernels/opt/conv2d.h>
#include <global/operator_factory.h>
#include <backend/name.h>

using namespace ts;
using namespace opt;
TS_REGISTER_OPERATOR(Conv2D, OPT, name::layer::conv2d())
