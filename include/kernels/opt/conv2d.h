#ifndef TENSORSTACK_KERNELS_OPT_CONV2D_H
#define TENSORSTACK_KERNELS_OPT_CONV2D_H

#include "operator_on_opt.h"
#include "backend/base/base_conv2d.h"
#include "conv2d_core.h"


namespace ts {
	namespace opt {
	    using Conv2D = base::Conv2DWithCore<OperatorOnOPT<base::Conv2D>, Conv2DCore>;
	}
}


#endif //TENSORSTACK_KERNELS_OPT_CONV2D_H