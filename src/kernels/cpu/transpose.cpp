#include <kernels/cpu/transpose.h>
#include <set>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>
#include <core/tensor_builder.h>

namespace ts {
    namespace cpu {

        template<typename T>
        static void Transpose_transpose_run(
                const T *psrc, T *pdst, int len,
                const std::vector<int> &permute,
                const Shape &input_shape, const Shape &output_shape) {
            Shape tmpshape(input_shape.size());

            // HypeShape hype_input_shape(input_shape);
            HypeShape hype_output_shape(output_shape);
            ShapeIterator input_shape_it(input_shape);

            int index = 0;
            for (int i = 0; i < len; i++) {
                auto &oldshape = input_shape_it.coordinate();
                for (size_t k = 0; k < oldshape.size(); k++) {
                    tmpshape[k] = oldshape[permute[k]];
                }

                index = hype_output_shape.to_index(tmpshape);//to_index(reshape, tmpshape);

                //if(index < 0) {
                //    throw ts::Exception("transpose operator failed, index is invalid");
                //}
                pdst[index] = psrc[i];
                ++input_shape_it;
            }
        }

        template<typename T>
        static inline void cpu_transpose_compute_run(const Tensor &x, const std::vector<int> &permute, Tensor &out) {
            Transpose_transpose_run(x.data<T>(), out.data<T>(), x.count(), permute, x.sizes(), out.sizes());
        }

        void Transpose::transpose(const Tensor &x, const std::vector<int> &permute, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_transpose_compute_run<TYPE>(x, permute, out); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << "add not support this data type: " << dtype << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Transpose, CPU, name::layer::transpose())
