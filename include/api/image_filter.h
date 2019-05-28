//
// Created by keir on 2019/3/16.
//

#ifndef TENSORSTACK_API_IMAGE_FILTER_H
#define TENSORSTACK_API_IMAGE_FILTER_H

#include "common.h"
#include "device.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Filter image before workbench run. Image should be NHWC or HWC format.
 * @see ts_Workbench_bind_filter and ts_Workbench_bind_filter_by_name
 */
struct ts_ImageFilter;
typedef struct ts_ImageFilter ts_ImageFilter;

enum ts_ResizeMethod {
    TS_RESIZE_BILINEAR = 0,
    TS_RESIZE_BICUBIC = 1,
    TS_RESIZE_NEAREST = 2,
};
typedef enum ts_ResizeMethod ts_ImageFilterResizeMethod;

/**
 * New ts_new_ImageFilter
 * @param device computing device, NULL means "CPU"
 * @return new ref of ts_ImageFilter
 * @note return NULL if failed
 * @note call @see ts_free_ImageFilter to free ts_ImageFilter
 */
TENSOR_STACK_C_API ts_ImageFilter *ts_new_ImageFilter(const ts_Device *device);

/**
 * Free ts_ImageFilter
 * @param filter the ts_ImageFilter ready to be free
 */
TENSOR_STACK_C_API void ts_free_ImageFilter(const ts_ImageFilter *filter);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_clear(ts_ImageFilter *filter);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_compile(ts_ImageFilter *filter);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_to_float(ts_ImageFilter *filter);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_scale(ts_ImageFilter *filter, float f);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_sub_mean(ts_ImageFilter *filter, const float *mean, int32_t len);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_div_std(ts_ImageFilter *filter, const float *std, int32_t len);

/**
 * @note using TS_RESIZE_BILINEAR by default
 */
TENSOR_STACK_C_API ts_bool ts_ImageFilter_resize(ts_ImageFilter *filter, int32_t width, int32_t height);

/**
 * @note using TS_RESIZE_BILINEAR by default
 */
TENSOR_STACK_C_API ts_bool ts_ImageFilter_resize_scalar(ts_ImageFilter *filter, int32_t width);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_center_crop(ts_ImageFilter *filter, int32_t width, int32_t height);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_channel_swap(ts_ImageFilter *filter, const int32_t *shuffle, int32_t len);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_to_chw(ts_ImageFilter *filter);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_prewhiten(ts_ImageFilter *filter);

/**
 * @note using TS_RESIZE_BILINEAR by default
 */
TENSOR_STACK_C_API ts_bool ts_ImageFilter_letterbox(ts_ImageFilter *filter, int32_t width, int32_t height, float outer_value);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_divided(ts_ImageFilter *filter, int32_t width, int32_t height, float padding_value);

TENSOR_STACK_C_API ts_Tensor *ts_ImageFilter_run(ts_ImageFilter *filter, const ts_Tensor *tensor);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_letterbox_v2(ts_ImageFilter *filter, int32_t width, int32_t height, float outer_value,
                                                       ts_ResizeMethod method);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_resize_v2(ts_ImageFilter *filter, int32_t width, int32_t height,
                                                    ts_ResizeMethod method);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_resize_scalar_v2(ts_ImageFilter *filter, int32_t width,
                                                           ts_ResizeMethod method);


#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_IMAGE_FILTER_H
