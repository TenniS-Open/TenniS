#include <kernels/cpu/add.h>
#include <core/tensor_builder.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <global/operator_factory.h>
#include <core/device.h>

#include <numeric>

namespace ts {
namespace cpu {


//////////////////////////////////////////////
Add::Add() {
}

static inline int to_mod_index(const HypeShape &hype, const std::vector<int> &coordinate) {
    auto temp = coordinate;
    for (size_t i = 0; i < temp.size(); ++i) {
        temp[i] %= hype.shape(i);
    }
    return hype.to_index(temp);
}

template<typename T>
static inline void compute_run(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
    HypeShape lhs_hype(lhs.sizes());
    HypeShape rhs_hype(rhs.sizes());
    HypeShape out_hype(out.sizes());

    auto plhs = lhs.data<T>();
    auto prhs = rhs.data<T>();
    auto pout = out.data<T>();

    auto ncount = out.count();
    for(int i = 0; i < ncount; i++) {
        std::vector<int> tmpshape = out_hype.to_coordinate(i);
        pout[i] = plhs[to_mod_index(lhs_hype, tmpshape)] + prhs[to_mod_index(rhs_hype, tmpshape)];
    }
}

template<typename T>
static inline void compute_run_scalar(const T *plhs, T scalar, T *pout, size_t count) {
    // this is CPU operator, so just using memcpy
    if (pout != plhs) std::memcpy(pout, plhs, count * sizeof(T));

    for (size_t i = 0; i < count; ++i) {
        pout[i] += scalar;
    }
}

template<typename T>
static inline void compute_run_same_shape(const T *plhs, const T *prhs, T *pout, size_t count) {
    // this is CPU operator, so just using memcpy
    if (pout != plhs) std::memcpy(pout, plhs, count * sizeof(T));

    for (size_t i = 0; i < count;++i) {
        pout[i] += prhs[i];
    }
}

template<typename T>
static inline void compute_run_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
    auto plhs = lhs.data<T>();
    auto prhs = rhs.data<T>();
    auto pout = out.data<T>();

    auto scalar = prhs[0];

    compute_run_scalar(plhs, scalar, pout, size_t(out.count()));
}


template<typename T>
static inline void compute_run_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
    auto plhs = lhs.data<T>();
    auto prhs = rhs.data<T>();
    auto pout = out.data<T>();

    compute_run_same_shape(plhs, prhs, pout, size_t(out.count()));
}


template<typename T>
static inline void compute_run_bias(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) {
    auto plhs = lhs.data<T>();
    auto prhs = rhs.data<T>();
    auto pout = out.data<T>();

    if (pout != plhs) std::memcpy(pout, plhs, out.count() * sizeof(T));

    auto &out_shape = out.sizes();

    auto number = std::accumulate(out_shape.begin(), out_shape.begin() + dim, 1, std::multiplies<int>());
    auto count = std::accumulate(out_shape.begin() + dim + 1, out_shape.end(), 1, std::multiplies<int>());

    auto channels = out_shape[dim];

    if (count == 1) {
        for (int n = 0; n < number; ++n) {
            auto pchannels = pout + n * channels;
            auto pscalar = prhs;
            for (int c = 0; c < channels; ++c) {
                *pchannels += *pscalar;
                ++pchannels;
                ++pscalar;
            }
        }
    } else {
        for (int n = 0; n < number; ++n) {
            for (int c = 0; c < channels; ++c) {
                int offset = (n * channels + c) * count;
                auto local_pout = pout + offset;
                compute_run_scalar(local_pout, prhs[channels], local_pout, size_t(count));
            }
        }
    }
}


void Add::reduce_with_broadcast(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
    // Notice: the all tensor' memory device are CPU, as given in running_memory_device
    DTYPE dtype = out.dtype();
    switch(dtype) {
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
            TS_LOG_ERROR << "add not support this data type: " << dtype << eject;
            break;
        }
    }
}

    void Add::reduce_with_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
        // Notice: the all tensor' memory device are CPU, as given in running_memory_device
        DTYPE dtype = out.dtype();
        switch(dtype) {
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
                TS_LOG_ERROR << "add not support this data type: " << dtype << eject;
                break;
            }
        }
    }

    void Add::reduce_with_bias(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) {
        // Notice: the all tensor' memory device are CPU, as given in running_memory_device
        DTYPE dtype = out.dtype();
        switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { compute_run_bias<TYPE>(lhs, rhs, out, dim); break; }
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

    void Add::reduce_with_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
        // Notice: the all tensor' memory device are CPU, as given in running_memory_device
        DTYPE dtype = out.dtype();
        switch(dtype) {
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
                TS_LOG_ERROR << "add not support this data type: " << dtype << eject;
                break;
            }
        }
    }

    MemoryDevice Add::running_memory_device() {
        return MemoryDevice(CPU);
    }

}
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Add, CPU, name::layer::add())

