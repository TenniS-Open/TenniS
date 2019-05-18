//
// Created by kier on 19-5-18.
//

#ifndef TENSORSTACK_API_OPERATOR_GPU_H
#define TENSORSTACK_API_OPERATOR_GPU_H

#include "operator.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <driver_types.h>
#define TS_CUDA_STREAM(stream) ((cudaStream_t)(stream))

/**
 *
 * @param op return value of @see ts_new_Operator
 * @param dict params dict
 */
typedef void ts_OperatorGPU_init(void *op, const ts_OperatorParams *dict, void *stream);

/**
 *
 * @param op return value of @see ts_new_Operator
 * @param argc argument count
 * @param argv argument values, are borrowed refs
 * @param stream CUDA stream, all kernel function must on stream
 * @return shape of ready values
 * infer return format int array tell the packed tensor shape
 * first element is fields_count, follow are each filed proto
 * for example, [2, TS_FLOAT32, 2, 4, 3, TS_INT32, 3, 5, 6, 7] means:
 *              {float32:[4, 3], int32:[5, 6, 7]}
 */
typedef ts_Tensor *ts_OperatorGPU_infer(void *op, int32_t argc, ts_Tensor **argv, void *stream);

/**
 *
 * @param op return value of @see ts_new_Operator
 * @param argc argument count
 * @param argv argument values, are borrowed refs
 * @param stream CUDA stream, all kernel function must on stream
 * @return packed values
 * @note return ts_new_Tensor_in_flow value
 */
typedef ts_Tensor *ts_OperatorGPU_run(void *op, int32_t argc, ts_Tensor **argv, void *stream);


TENSOR_STACK_C_API void ts_OperatorGPU_Register(
        const char *op,
        ts_new_Operator *f_new,
        ts_free_Operator *f_free,
        ts_OperatorGPU_init *f_init,
        ts_OperatorGPU_infer *f_infer,
        ts_OperatorGPU_run *f_run);

#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_OPERATOR_GPU_H
