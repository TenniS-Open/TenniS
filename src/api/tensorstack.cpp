//
// Created by kier on 2019/1/26.
//

#include "api/tensorstack.h"

#include <string>
#include <api/tensorstack.h>


#include "global/setup.h"
#include "utils/except.h"
#include "core/tensor.h"
#include "core/tensor_builder.h"

static thread_local std::string _thread_local_last_error_message;

using namespace ts;

#define DECLARE_API_TYPE(API_TYPE, TS_TYPE) \
struct API_TYPE { \
    using self = API_TYPE; \
    template <typename... Args> \
    explicit API_TYPE(Args &&...args) { \
        this->pointer = std::make_shared<TS_TYPE>(std::forward<Args>(args)...); \
    } \
    std::shared_ptr<TS_TYPE> pointer; \
    const TS_TYPE *operator->() const { return pointer.get(); } \
    TS_TYPE *operator->() { return pointer.get(); } \
    const TS_TYPE &operator*() const { return *pointer; } \
    TS_TYPE &operator*() { return *pointer; } \
    const TS_TYPE *get() const { return pointer.get(); } \
    TS_TYPE *get() { return pointer.get(); } \
};

DECLARE_API_TYPE(ts_Tensor, Tensor)

#define ts_false 0
#define ts_true 1

#define TRY_HEAD \
_thread_local_last_error_message.clear(); \
try {

#define RETURN_OR_CATCH(ret, cat) \
return ret; \
} catch (const Exception &e) { \
_thread_local_last_error_message = e.what(); \
return cat; \
}

#define TRY_TAIL \
} catch (const Exception &) { \
}

ts_bool ts_setup() {
    TRY_HEAD
        setup();
    RETURN_OR_CATCH(ts_true, ts_false);
}

const char *ts_last_error_message() {
    return _thread_local_last_error_message.c_str();
}

ts_Tensor *ts_new_Tensor(int32_t *shape, int32_t shape_len, ts_DTYPE dtype, const void *data) {
    TRY_HEAD
        std::unique_ptr<ts_Tensor> tensor(new ts_Tensor());

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
        if (!tensor) throw Exception("NullPointerException");
        const int32_t *shape = (*tensor)->sizes().data();
    RETURN_OR_CATCH(shape, nullptr)
}

int32_t ts_Tensor_shape_size(ts_Tensor *tensor) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException");
        auto size = int32_t((*tensor)->dims());
    RETURN_OR_CATCH(size, 0)
}

ts_DTYPE ts_Tensor_dtype(ts_Tensor *tensor) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException");
        auto dtype = ts_DTYPE((*tensor)->proto().dtype());
    RETURN_OR_CATCH(dtype, TS_VOID)
}

void *ts_Tensor_data(ts_Tensor *tensor) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException");
        auto data = (*tensor)->data();
    RETURN_OR_CATCH(data, nullptr)
}

ts_Tensor *ts_Tensor_clone(ts_Tensor *tensor) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor);
        dolly->pointer = (*tensor)->clone_shared();
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_bool ts_Tensor_sync_cpu(ts_Tensor *tensor) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException");
        (*tensor)->sync(MemoryDevice(CPU));
    RETURN_OR_CATCH(true, false)
}






