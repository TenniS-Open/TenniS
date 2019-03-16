//
// Created by keir on 2019/3/16.
//

#ifndef TENSORSTACK_API_WORKBENCH_H
#define TENSORSTACK_API_WORKBENCH_H

#include "common.h"
#include "tensor.h"
#include "module.h"
#import "image_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ts_Workbench;
typedef struct ts_Workbench ts_Workbench;

// Workbench API

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Workbench *ts_Workbench_Load(ts_Module *module, const ts_Device *device);

/**
 * Happen nothing if failed.
 */
TENSOR_STACK_C_API void ts_free_Workbench(const ts_Workbench *workbench);

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Workbench *ts_Workbench_clone(ts_Workbench *workbench);

/**
 * Return false if failed.
 */
TENSOR_STACK_C_API ts_bool ts_Workbench_input(ts_Workbench *workbench, int32_t i, const ts_Tensor *tensor);

/**
 * Return false if failed.
 */
TENSOR_STACK_C_API ts_bool ts_Workbench_input_by_name(ts_Workbench *workbench, const char *name, const ts_Tensor *tensor);

/**
 * Return false if failed.
 */
TENSOR_STACK_C_API ts_bool ts_Workbench_run(ts_Workbench *workbench);

/**
 * Return false if failed.
 */
TENSOR_STACK_C_API ts_bool ts_Workbench_output(ts_Workbench *workbench, int32_t i, ts_Tensor *tensor);

/**
 * Return false if failed.
 */
TENSOR_STACK_C_API ts_bool ts_Workbench_output_by_name(ts_Workbench *workbench, const char *name, ts_Tensor *tensor);

/**
 * Happen Nothing if failed.
 */
TENSOR_STACK_C_API ts_bool ts_Workbench_set_computing_thread_number(ts_Workbench *workbench, int32_t number);

TENSOR_STACK_C_API ts_bool ts_Workbench_bind_filter(ts_Workbench *workbench, int32_t i, ts_ImageFilter *filter);

TENSOR_STACK_C_API ts_bool ts_Workbench_bind_filter_by_name(ts_Workbench *workbench, const char *name, ts_ImageFilter *filter);



#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_WORKBENCH_H
