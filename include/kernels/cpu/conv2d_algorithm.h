#include <core/tensor_builder.h>
#include <kernels/cpu/math_cpu.h>
#include <kernels/cblas/math_cblas.h>
#include "runtime/inside/thread_pool.h"
#include "utils/ctxmgr_lite.h"
#include "utils/box.h"

#include "core/tensor.h"

namespace ts {
    namespace cpu {
        template<typename T>
        class TS_DEBUG_API Conv2dAlgorithm {
        public:
           static void conv3x3_winograd23_transform_kernel(const Tensor& kernel, Tensor &kernel_tm);

           static void conv3x3_winograd23_transform_kernel_inplace(const Tensor& kernel, Tensor &kernel_tm);

           static void conv3x3_winograd63_transform_kernel(const Tensor& kernel, Tensor &kernel_tm);

           static void conv3x3_winograd63_transform_kernel_inplace(const Tensor& kernel, Tensor &kernel_tm);

           static void conv3x3_winograd23_threadpool(const Tensor &x, const Tensor &k_tm, Tensor &out);

           static void conv3x3_winograd23(const Tensor &x, const Tensor &k_tm, Tensor &out);

           static void conv3x3_winograd63_threadpool(const Tensor &x, const Tensor &k_tm, Tensor &out);

           static void conv3x3_winograd63(const Tensor &x, const Tensor &w, Tensor &out);

           static void conv2d_3x3_sse(const Tensor &x, const Tensor &w, Tensor &out);

           static void conv2d_3x3_sse_inplace(const Tensor &x, const Tensor &w, Tensor &out);
        };
    }

}

//extern template class ts::opt::Conv2dAlgorithm<ts::dtype<ts::FLOAT32>::declare>;
//extern template class ts::opt::Conv2dAlgorithm<ts::dtype<ts::FLOAT64>::declare>;
