//
// Created by kier on 2019/3/16.
//

#include "declare_tensor.h"

#include "core/tensor_builder.h"

using namespace ts;

ts_Tensor *ts_new_Tensor(int32_t *shape, int32_t shape_len, ts_DTYPE dtype, const void *data) {
    TRY_HEAD
    if (shape == nullptr) shape_len = 0;
    std::unique_ptr<ts_Tensor> tensor(new ts_Tensor());

    if (data == nullptr) {
        **tensor = Tensor(DTYPE(dtype), Shape(shape, shape + shape_len));
        return tensor.release();
    }

    switch (dtype) {
        default:
            TS_LOG_ERROR << "Not support dtype: " << dtype << eject;
            break;
#define DECLARE_TENSOR_BUILD(api_dtype, type) \
        case api_dtype: \
            **tensor = tensor::build(DTYPE(dtype), Shape(shape, shape + shape_len), \
                    reinterpret_cast<const type*>(data)); \
            break;

        DECLARE_TENSOR_BUILD(TS_INT8, int8_t)
        DECLARE_TENSOR_BUILD(TS_UINT8, uint8_t)
        DECLARE_TENSOR_BUILD(TS_INT16, int16_t)
        DECLARE_TENSOR_BUILD(TS_UINT16, uint16_t)
        DECLARE_TENSOR_BUILD(TS_INT32, int32_t)
        DECLARE_TENSOR_BUILD(TS_UINT32, uint32_t)
        DECLARE_TENSOR_BUILD(TS_INT64, int64_t)
        DECLARE_TENSOR_BUILD(TS_UINT64, uint64_t)
        DECLARE_TENSOR_BUILD(TS_FLOAT32, float)
        DECLARE_TENSOR_BUILD(TS_FLOAT64, double)

#undef DECLARE_TENSOR_BUILD
    }

    RETURN_OR_CATCH(tensor.release(), nullptr)
}

void ts_free_Tensor(const ts_Tensor *tensor) {
    TRY_HEAD
    delete tensor;
    TRY_TAIL
}

const int32_t *ts_Tensor_shape(ts_Tensor *tensor) {
    TRY_HEAD
    if (!tensor) throw Exception("NullPointerException: @param: 1");
    const int32_t *shape = (*tensor)->sizes().data();
    RETURN_OR_CATCH(shape, nullptr)
}

int32_t ts_Tensor_shape_size(ts_Tensor *tensor) {
    TRY_HEAD
    if (!tensor) throw Exception("NullPointerException: @param: 1");
    auto size = int32_t((*tensor)->dims());
    RETURN_OR_CATCH(size, 0)
}

ts_DTYPE ts_Tensor_dtype(ts_Tensor *tensor) {
    TRY_HEAD
    if (!tensor) throw Exception("NullPointerException: @param: 1");
    auto dtype = ts_DTYPE((*tensor)->proto().dtype());
    RETURN_OR_CATCH(dtype, TS_VOID)
}

void *ts_Tensor_data(ts_Tensor *tensor) {
    TRY_HEAD
    if (!tensor) throw Exception("NullPointerException: @param: 1");
    auto data = (*tensor)->data();
    RETURN_OR_CATCH(data, nullptr)
}

ts_Tensor *ts_Tensor_clone(ts_Tensor *tensor) {
    TRY_HEAD
    if (!tensor) throw Exception("NullPointerException: @param: 1");
    std::unique_ptr<ts_Tensor> dolly(new ts_Tensor((*tensor)->clone_shared()));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_bool ts_Tensor_sync_cpu(ts_Tensor *tensor) {
    TRY_HEAD
    if (!tensor) throw Exception("NullPointerException: @param: 1");
    (*tensor)->sync(MemoryDevice(CPU));
    RETURN_OR_CATCH(true, false)
}

