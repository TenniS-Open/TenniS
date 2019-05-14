//
// Created by keir on 2019/3/16.
//

#ifndef TENSORSTACK_API_TENSOR_H
#define TENSORSTACK_API_TENSOR_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ts_Tensor;
typedef struct ts_Tensor ts_Tensor;

enum ts_DTYPE {
    TS_VOID        = 0,
    TS_INT8        = 1,
    TS_UINT8       = 2,
    TS_INT16       = 3,
    TS_UINT16      = 4,
    TS_INT32       = 5,
    TS_UINT32      = 6,
    TS_INT64       = 7,
    TS_UINT64      = 8,
    TS_FLOAT32     = 10,
    TS_FLOAT64     = 11,
    TS_CHAR8       = 13,
};
typedef enum ts_DTYPE ts_DTYPE;

enum ts_InFlow {
    TS_HOST     = 0,
    TS_DEVICE   = 1,
};
typedef enum ts_InFlow ts_InFlow;

// Tensor's API

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Tensor *ts_new_Tensor(const int32_t *shape, int32_t shape_len, ts_DTYPE dtype, const void *data);

/**
 * Happen nothing if failed.
 */
TENSOR_STACK_C_API void ts_free_Tensor(const ts_Tensor *tensor);

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API const int32_t *ts_Tensor_shape(ts_Tensor *tensor);

/**
 * Return zero if failed.
 */
TENSOR_STACK_C_API int32_t ts_Tensor_shape_size(ts_Tensor *tensor);

/**
 * Return TS_VOID if failed.
 */
TENSOR_STACK_C_API ts_DTYPE ts_Tensor_dtype(ts_Tensor *tensor);

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API void* ts_Tensor_data(ts_Tensor *tensor);

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Tensor *ts_Tensor_clone(ts_Tensor *tensor);

/**
 * Return false if failed.
 */
TENSOR_STACK_C_API ts_bool ts_Tensor_sync_cpu(ts_Tensor *tensor);

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Tensor *ts_Tensor_cast(ts_Tensor *tensor, ts_DTYPE dtype);

/**
 * Return NULL if failed.
 * Return new tensor.
 */
TENSOR_STACK_C_API ts_Tensor *ts_Tensor_reshape(ts_Tensor *tensor, const int32_t *shape, int32_t shape_len);

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Tensor *ts_new_Tensor_in_flow(ts_InFlow in_flow, const int32_t *shape, int32_t shape_len, ts_DTYPE dtype, const void *data);

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Tensor *ts_Tensor_view_in_flow(ts_Tensor *tensor, ts_InFlow in_flow);


/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Tensor *ts_Tensor_field(ts_Tensor *tensor, int32_t index);


/**
 * Return false if failed. False also mean no packed
 */
TENSOR_STACK_C_API ts_bool ts_Tensor_packed(ts_Tensor *tensor);


/**
 * Return 0 if failed.
 */
TENSOR_STACK_C_API int32_t ts_Tensor_fields_count(ts_Tensor *tensor);


/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Tensor *ts_Tensor_pack(ts_Tensor **fields, int32_t count);


/**
 * Return NULL if failed
 */
TENSOR_STACK_C_API ts_Tensor *ts_Tensor_slice(ts_Tensor *tensor, int32_t i);


/**
 * Return NULL if failed
 */
TENSOR_STACK_C_API ts_Tensor *ts_Tensor_slice_v2(ts_Tensor *tensor, int32_t beg, int32_t end);


#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_TENSOR_H
