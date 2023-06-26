//
// Created by kier on 2019/3/6.
//

#include "kernels/cpu/operator_on_cpu.h"
#include "backend/base/base_gather_elements.h"
#include "global/operator_factory.h"
#include "backend/name.h"

#include <numeric>

namespace ts {
    namespace cpu {
        class GatherElements : public OperatorOnAny<base::GatherElements> {
        public:
            using self = GatherElements;
            using supper = OperatorOnCPU<base::GatherElements>;

            void gather(const Tensor &x, const Tensor &indices, int axis, Tensor &out) override;
            void gather_cpu(const Tensor &x, const Tensor &indices, int axis, Tensor &out);
        };

        template <typename T>
        void compute_run(const Tensor &x, const Tensor &indices, int axis, Tensor &out) {
            auto &x_shape = x.sizes();
            auto number = std::accumulate(x_shape.begin(), x_shape.begin() + axis, 1,
                                          std::multiplies<int>());
            auto width = std::accumulate(x_shape.begin() + axis + 1, x_shape.end(), 1,
                                         std::multiplies<int>());

            HypeShape norm_x_shape({number, x_shape[axis], width});
            HypeShape norm_out_shape({number, width});

            auto x_data = x.data<T>();
            auto out_data = out.data<T>();
            auto indices_data = indices.data<int32_t>();

            std::vector<int> norm_x_coord = {0, 0, 0};
            std::vector<int> norm_out_coord {0, 0};

            for (int n = 0; n < number; ++n) {
                norm_out_coord[0] = n;
                norm_x_coord[0] = n;
                for (int i = 0; i < width; ++i) {
                    norm_out_coord[1] = i;
                    norm_x_coord[2] = i;

                    auto out_index = norm_out_shape.to_index(norm_out_coord);
                    norm_x_coord[1] = indices_data[out_index];
                    auto x_index = norm_x_shape.to_index(norm_x_coord);

                    out_data[out_index] = x_data[x_index];
                }
            }
        }

        void GatherElements::gather_cpu(const Tensor &x, const Tensor &indices, int axis, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { compute_run<TYPE>(x, indices, axis, out); break; }
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
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }

        void GatherElements::gather(const Tensor &x, const Tensor &indices, int axis, Tensor &out) {
            if (out.device().type() == CPU && x.device().type() == CPU) {
                gather_cpu(x, indices, axis, out);
                return;
            }
            auto memcpy_handler = HardConverter::Query(out.device().type(), x.device().type());
            TS_AUTO_CHECK(memcpy_handler != nullptr);
            auto device_id = out.device().id();

            auto &x_shape = x.sizes();
            auto number = std::accumulate(x_shape.begin(), x_shape.begin() + axis, 1, std::multiplies<int>());
            auto width = std::accumulate(x_shape.begin() + axis + 1, x_shape.end(), 1, std::multiplies<int>());

            HypeShape norm_x_shape({number, x_shape[axis], width});
            HypeShape norm_out_shape({number, width});

            auto bytes = x.proto().type_bytes();

            auto x_data = x.data<char>();
            auto out_data = out.data<char>();
            auto indices_data = indices.data<int32_t>();

            std::vector<int> norm_x_coord = {0, 0, 0};
            std::vector<int> norm_out_coord {0, 0};

            for (int n = 0; n < number; ++n) {
                norm_out_coord[0] = n;
                norm_x_coord[0] = n;
                for (int i = 0; i < width; ++i) {
                    norm_out_coord[1] = i;
                    norm_x_coord[2] = i;

                    auto out_index = norm_out_shape.to_index(norm_out_coord);
                    norm_x_coord[1] = indices_data[out_index];
                    auto x_index = norm_x_shape.to_index(norm_x_coord);

                    auto src_ptr = x_data + x_index * bytes;
                    auto dst_ptr = out_data + out_index * bytes;
                    memcpy_handler(device_id, dst_ptr, device_id, src_ptr, size_t(bytes));
                }
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(GatherElements, CPU, "gather_elements")
