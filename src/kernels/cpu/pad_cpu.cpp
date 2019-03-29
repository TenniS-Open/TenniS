#include <kernels/cpu/pad_cpu.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

#include <core/memory.h>
#include <numeric>

namespace ts {
    namespace cpu {
        /**
         *
         * @param [in] out_coord the coordinate in output map
         * @param [in] padding padding
         * @param [in] x_shape the input shape
         * @param [out] x_coord the transpose back coordinate
         * @return true if transposed, else return false
         */
        static inline bool transpose_pad(const std::vector<int> &out_coord,
                const std::vector<std::array<int, 2>> &padding,
                const std::vector<int> &x_shape,
                std::vector<int> &x_coord) {
            auto dims = out_coord.size();
            x_coord.resize(dims);
            for (size_t i = 0; i < dims; ++i) {
                auto offset = out_coord[i] - padding[i][0];
                if (offset < 0 || offset >= x_shape[i]) return false;
                x_coord[i] = offset;
            }
            return true;
        }

        template <typename T>
        static inline void cpu_pad_compute_run(const Tensor &x, const std::vector<std::array<int, 2>> &padding, float padding_value, Tensor &out) {
            ShapeIterator out_it(out.sizes());
            auto &x_shape = x.sizes();
            std::vector<int> x_coord;
            auto out_data = out.data<T>();
            auto x_data = x.data<T>();

            HypeShape hype_x_shape(x_shape);

            auto value = T(padding_value);

            int count = out.count();
            for (int i = 0; i < count; ++i) {
                bool not_overflow = transpose_pad(out_it.coordinate(), padding, x_shape, x_coord);
                if (not_overflow) {
                    out_data[i] = x_data[hype_x_shape.to_index(x_coord)];
                } else {
                    out_data[i] = value;
                }
                ++out_it;
            }
        }

        void PadOnCPU::pad(const Tensor &x, const std::vector<std::array<int, 2>> &padding, float padding_value, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_pad_compute_run<TYPE>(x, padding, padding_value, out); break; }
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
                    TS_LOG_ERROR << "pad not support this data type: " << dtype << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(PadOnCPU, CPU, name::layer::pad())
