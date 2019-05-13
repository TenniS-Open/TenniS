//
// Created by kier on 19-5-8.
//

#ifndef TENSORSTACK_API_OPERATOR_H
#define TENSORSTACK_API_OPERATOR_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ts_OperatorParams;
typedef struct ts_OperatorParams ts_OperatorParams;

/**
 * Return new reference, NULL if param is not exist
 */
TENSOR_STACK_C_API ts_Tensor *ts_OperatorParams_get(const ts_OperatorParams *dict, const char *param);

/**
 *
 * @return new object of operator
 */
typedef void *ts_new_Operator();

/**
 *
 * @param op return value of @see ts_new_Operator
 */
typedef void ts_free_Operator(const void *op);

/**
 *
 * @param op return value of @see ts_new_Operator
 * @param dict params dict
 */
typedef void ts_Operator_init(void *op, const ts_OperatorParams *dict);

/**
 *
 * @param op return value of @see ts_new_Operator
 * @param argc argument count
 * @param argv argument values, are borrowed refs
 * @return shape of ready values
 * infer return format int array tell the packed tensor shape
 * first element is fields_count, follow are each filed proto
 * for example, [2, TS_FLOAT32, 2, 4, 3, TS_INT32, 3, 5, 6, 7] means:
 *              {float32:[4, 3], int32:[5, 6, 7]}
 */
typedef ts_Tensor *ts_Operator_infer(void *op, int32_t argc, ts_Tensor **argv);

/**
 *
 * @param op return value of @see ts_new_Operator
 * @param argc argument count
 * @param argv argument values, are borrowed refs
 * @return packed values
 * @note return ts_new_Tensor_in_flow value
 */
typedef ts_Tensor *ts_Operator_run(void *op, int32_t argc, ts_Tensor **argv);


TENSOR_STACK_C_API void ts_Operator_Register(
        const char *device, const char *op,
        ts_new_Operator *f_new,
        ts_free_Operator *f_free,
        ts_Operator_init *f_init,
        ts_Operator_infer *f_infer,
        ts_Operator_run *f_run);


TENSOR_STACK_C_API void ts_Operator_Throw(const char *message);

TENSOR_STACK_C_API void ts_Operator_ThrowV2(const char *message, const char *filename, int32_t line_number);

#define TS_C_THROW(message) \
    ts_Operator_ThrowV2((message), __FILE__, __LINE__)

#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_OPERATOR_H
