#include "roi_align_op.h"
#include "core/ieee754_float.h"

#include "op_kernel.h"

namespace ts {

namespace dragon {

template <class Context> template <typename T>
void ROIAlignOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Rdata = Input(1).template data<float, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::ROIAlign(
        Input(0).dim(1), Input(0).dim(2), Input(0).dim(3),
            pool_h, pool_w, Input(1).dim(0),
                spatial_scale, sampling_ratio,
                    Xdata, Rdata, Ydata, ctx());
}

template <class Context>
void ROIAlignOp<Context>::RunOnDevice() {
    Output(0)->Reshape({
        Input(1).dim(0),    /*!   Number of RoIs  */
        Input(0).dim(1),    /*!   Channels        */
        pool_h,             /*!   Pooled height   */
        pool_w              /*!   Pooled width    */
    });

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" }) << eject;
}

}  // namespace dragon

}  // namespace ts

template class ts::dragon::ROIAlignOp<ts::dragon::CPUContext>;
#ifdef TS_USE_CUDA
template class ts::dragon::ROIAlignOp<ts::dragon::CUDAContext>;
#endif