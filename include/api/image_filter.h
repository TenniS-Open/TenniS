//
// Created by keir on 2019/3/16.
//

#ifndef TENSORSTACK_API_IMAGE_FILTER_H
#define TENSORSTACK_API_IMAGE_FILTER_H

#include "common.h"
#include "device.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ts_ImageFilter;
typedef struct ts_ImageFilter ts_ImageFilter;

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_ImageFilter *ts_new_ImageFilter(const ts_Device *device);

/**
 * Happen nothing if failed.
 */
TENSOR_STACK_C_API void ts_free_ImageFilter(const ts_ImageFilter *filter);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_clear(ts_ImageFilter *filter);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_compile(ts_ImageFilter *filter);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_to_float(ts_ImageFilter *filter);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_scale(ts_ImageFilter *filter, float f);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_sub_mean(ts_ImageFilter *filter, const float *mean, int32_t len);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_div_std(ts_ImageFilter *filter, const float *std, int32_t len);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_resize(ts_ImageFilter *filter, int width, int height);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_resize_scalar(ts_ImageFilter *filter, int width);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_center_crop(ts_ImageFilter *filter, int width, int height);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_channel_swap(ts_ImageFilter *filter, const int *shuffle, int32_t len);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_to_chw(ts_ImageFilter *filter);


#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_IMAGE_FILTER_H
