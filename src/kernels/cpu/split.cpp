#include <kernels/cpu/split.h>
#include <core/tensor_builder.h>

#include <numeric>

#include "global/operator_factory.h"
#include "global/hard_converter.h"

namespace ts {
    namespace cpu {
        template <typename T>
        static void cpu_split_compute_run(const Tensor &x, const std::vector<int> &split, int dim,
                                          std::vector<Tensor> &out) {
            auto memcpy = HardConverter::Query(x.device().type(), x.device().type());
            TS_AUTO_CHECK(memcpy != nullptr);
            auto device_id = x.device().id();

            auto shape = x.sizes();
            auto N = std::accumulate(shape.begin(), shape.begin() + dim, 1, std::multiplies<int32_t>());
            auto W = std::accumulate(shape.begin() + dim + 1, shape.end(), 1, std::multiplies<int32_t>());

            auto S = split.size();

            auto from = x.data<T>();
            std::vector<T*> tos(split.size());
            for (decltype(S) i = 0; i < S; ++i) {
                tos[i] = out[i].data<T>();
            }

            for (decltype(N) n = 0; n < N; ++n) {
                for (decltype(S) i = 0; i < S; ++i) {
                    auto s = split[i];
                    auto &to = tos[i];

                    // copy data from `from` to `to`
                    auto count = s * W;
                    auto bytes = count * type_bytes(x.dtype());
                    memcpy(device_id, to, device_id, from, size_t(bytes));

                    from += count;
                    to += count;
                }
            }
        }
        void SplitOp::split(const Tensor &x, const std::vector<int> &split, int dim,
                          std::vector<Tensor> &out) {
            DTYPE dtype = x.dtype();

            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_split_compute_run<TYPE>(x, split, dim, out); break; }
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
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(SplitOp, ts::CPU, "split")
