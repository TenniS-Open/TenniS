#include "kernels/cpu/equal.h"
#include "global/operator_factory.h"
#include <algorithm>
#include "kernels/common/simd.h"

namespace ts {
    namespace cpu {
        template<typename T>
        static inline void reduce_operator(T &x, T lhs, T rhs) {
            x = lhs == rhs;
        }

        template<typename T>
        static inline void reduce_operator(T &x, T y) {
            x = x == y;
        }

        static inline int to_mod_index(const HypeShape &hype, const Shape &coordinate) {
            auto temp = coordinate;
            for (size_t i = 0; i < temp.size(); ++i) {
                temp[i] %= hype.shape(i);
            }
            return hype.to_index(temp);
        }

        template<typename T>
        static inline void compute_run(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            HypeShape lhs_type(lhs.sizes());
            HypeShape rhs_type(rhs.sizes());

            ShapeIterator out_iterator(out.sizes());

            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();

            auto ncount = out.count();
            for (int i = 0; i < ncount; ++i) {
                auto &tmpshape = out_iterator.coordinate();
                reduce_operator(pout[i], plhs[to_mod_index(lhs_type, tmpshape)],
                                prhs[to_mod_index(rhs_type, tmpshape)]);
                ++out_iterator;
            }
        }

        template<typename T>
        static inline void compute_run_scalar(const T *plhs, T scalar, T *pout, int count) {
            // this is CPU operator, so just using memcpy
            if (pout != plhs) std::memcpy(pout, plhs, count * sizeof(T));

            for (int i = 0; i < count; ++i) {
                reduce_operator(pout[i], scalar);
            }
        }

        template<>
        inline void compute_run_scalar(const float *plhs, float scalar, float *pout, int count) {
            float32x4 scalarx4(scalar);
            for (int i = 0; i < count - 3; i += 4) {
                float32x4 pout_x4 = float32x4(&plhs[i]) == scalarx4;
                pout_x4.store(&pout[i]);
            }
            for (int i = count / 4 * 4; i < count; ++i) {
                reduce_operator(pout[i], plhs[i], scalar);
            }
        }

        template<typename T>
        static inline void compute_run_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();

            auto scalar = prhs[0];

            compute_run_scalar(plhs, scalar, pout, out.count());
        }

        template<typename T>
        static inline void compute_run_same_shape(const T *plhs, const T *prhs, T *pout, int count) {
            if (pout != plhs) std::memcpy(pout, plhs, count * sizeof(T));

            for (int i = 0; i < count; ++i) {
                reduce_operator(pout[i], prhs[i]);
            }
        }

        template<>
        inline void compute_run_same_shape(const float *plhs, const float *prhs, float *pout, int count) {
            for (int i = 0; i < count - 3; i += 4) {
                float32x4 pout_x4 = float32x4(&plhs[i]) == float32x4(&prhs[i]);
                pout_x4.store(&pout[i]);
            }

            for (int i = count / 4 * 4; i < count; ++i) {
                reduce_operator(pout[i], plhs[i], prhs[i]);
            }
        }

        template<typename T>
        void compute_run_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();

            compute_run_same_shape(plhs, prhs, pout, out.count());
        }

        void Equal::reduce_with_broadcast(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            DTYPE dtype = lhs.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { compute_run<TYPE>(lhs, rhs, out); break; }
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

        void Equal::reduce_with_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            DTYPE dtype = lhs.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { compute_run_scalar<TYPE>(lhs, rhs, out); break; }
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

        void Equal::reduce_with_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = lhs.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { compute_run_same_shape<TYPE>(lhs, rhs, out); break; }
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

        void Equal::reduce_with_scalar_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            this->reduce_with_scalar(rhs, lhs, out);
        }

    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Equal, ts::CPU, "equal")
