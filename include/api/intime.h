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

TENSOR_STACK_C_API ts_Tensor *ts_intime_transpose(const ts_Tensor *x, const int32_t *permute, int32_t len);

TENSOR_STACK_C_API ts_Tensor *ts_intime_sigmoid(const ts_Tensor *x);

#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_INTIME_H
