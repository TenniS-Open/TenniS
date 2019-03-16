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
};
typedef enum ts_DTYPE ts_DTYPE;

// Tensor's API

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Tensor *ts_new_Tensor(int32_t *shape, int32_t shape_len, ts_DTYPE dtype, const void *data);

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

#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_TENSOR_H
