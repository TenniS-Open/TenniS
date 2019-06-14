//
// Created by kier on 19-5-13.
//

#ifndef TENSORSTACK_API_INTIME_H
#define TENSORSTACK_API_INTIME_H

#include "common.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Return transposed x with permute. `out = transpose(x, axes=permute)`.
 * @param x input tensor
 * @param permute dest tensor axes
 * @param len length of permute
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 */
TENSOR_STACK_C_API ts_Tensor *ts_intime_transpose(const ts_Tensor *x, const int32_t *permute, int32_t len);

/**
 * Return sigmoid x. `out = 1 / (1 + e^{-x})`
 * @param x input tensor
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 */
TENSOR_STACK_C_API ts_Tensor *ts_intime_sigmoid(const ts_Tensor *x);

/**
 * Return gathered x. `out = take(x, indices=indices, axis=axis)`
 * @param x input tensor
 * @param indices indices
 * @param axis axis
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 */
TENSOR_STACK_C_API ts_Tensor *ts_intime_gather(const ts_Tensor *x, const ts_Tensor *indices, int32_t axis);

/**
 * Return concat x. `out = concat(x, axis=dim)`
 * @param x input tensors
 * @param len length of input tensors
 * @param dim concat dim
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 */
TENSOR_STACK_C_API ts_Tensor *ts_intime_concat(const ts_Tensor *const *x, int32_t len, int32_t dim);

/**
 * Return softmax x. output y.
 * @param x input tensor
 * @param dim softmax on given dim
 * @param smooth if smooth mode
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 * if not smooth:
 * ```
 * y_i = exp(x_i) / \sum{exp(x_i)}
 * ```
 * else:
 * ```
 * t_i = x_i - max(x)
 * y_i = exp(t_i) / \sum{exp(t_i)}
 * in framework like caffe, smooth is true.
 */
TENSOR_STACK_C_API ts_Tensor *ts_intime_softmax(const ts_Tensor *x, int32_t dim, ts_bool smooth);

/**
 * Return pad x.
 * @param x input tensor
 * @param padding Int[_, 2] than first axis must equal to dims of x
 * @param padding_value padding value
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 * @note padding size can be neg value.
 */
TENSOR_STACK_C_API ts_Tensor *ts_intime_pad(const ts_Tensor *x, const ts_Tensor *padding, float padding_value);

/**
 * Return given dtype tensor.
 * @param x input tensor
 * @param dtype @sa ts_DTYPE
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 *
 */
TENSOR_STACK_C_API ts_Tensor *ts_intime_cast(const ts_Tensor *x, ts_DTYPE dtype);

#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_INTIME_H
