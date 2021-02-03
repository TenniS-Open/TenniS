#include <kernels/cpu/where.h>
#include "backend/name.h"
#include "global/operator_factory.h"

namespace ts {
    namespace cpu {
        template<typename T>
        static inline void reduce_operator(T &x, uint8_t cond, T lhs, T rhs) {
            x = cond ? lhs : rhs;
        }

        static inline int to_mod_index(const HypeShape &hype, const Shape &coordinate) {
            auto temp = coordinate;
            for (size_t i = 0; i < temp.size(); ++i) {
                temp[i] %= hype.shape(i);
            }
            return hype.to_index(temp);
        }

        template<typename T>
        static inline void
        compute_run_broadcast(const Tensor &cond, const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            HypeShape cond_hype(cond.sizes());
            HypeShape lhs_hype(lhs.sizes());
            HypeShape rhs_hype(rhs.sizes());
            ShapeIterator out_iterator(out.sizes());

            auto pcond = cond.data<uint8_t>();
            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();

            auto ncount = out.count();
            for (int i = 0; i < ncount; ++i) {
                auto &tmpshape = out_iterator.coordinate();
                reduce_operator(pout[i], pcond[to_mod_index(cond_hype, tmpshape)],
                                plhs[to_mod_index(lhs_hype, tmpshape)], prhs[to_mod_index(rhs_hype, tmpshape)]);
                ++out_iterator;
            }
        }

        template<typename T>
        static inline void
        compute_run_same_shape(const Tensor &cond, const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            auto pcond = cond.data<uint8_t>();
            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();

            auto ncount = out.count();
            for (int i = 0; i < ncount; ++i) {
                pout[i] = pcond[i] ? plhs[i] : prhs[i];
            }
        }

        void Where::reduce_with_broadcast(const Tensor &cond, const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { compute_run_broadcast<TYPE>(cond, lhs, rhs, out); break; }
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
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                 << eject;
                    break;
                }
            }
        }

        void Where::reduce_with_same_shape(const Tensor &cond, const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { compute_run_same_shape<TYPE>(cond, lhs, rhs, out); break; }
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
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                 << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Where, ts::CPU, "where")
