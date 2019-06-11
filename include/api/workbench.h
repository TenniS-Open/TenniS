//
// Created by keir on 2019/3/16.
//

#ifndef TENSORSTACK_API_WORKBENCH_H
#define TENSORSTACK_API_WORKBENCH_H

#include "common.h"
#include "tensor.h"
#include "module.h"
#include "image_filter.h"
#include "program.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ts_Workbench;
typedef struct ts_Workbench ts_Workbench;

// Workbench API

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Workbench *ts_Workbench_Load(const ts_Module *module, const ts_Device *device);

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

TENSOR_STACK_C_API ts_bool ts_Workbench_bind_filter(ts_Workbench *workbench, int32_t i, const ts_ImageFilter *filter);

TENSOR_STACK_C_API ts_bool ts_Workbench_bind_filter_by_name(ts_Workbench *workbench, const char *name, const ts_ImageFilter *filter);

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Workbench *ts_new_Workbench(const ts_Device *device);

/**
 * Return false if failed.
 */
TENSOR_STACK_C_API ts_bool ts_Workbench_setup(ts_Workbench *workbench, const ts_Program *program);

/**
 * Return false if failed.
 */
TENSOR_STACK_C_API ts_bool ts_Workbench_setup_context(ts_Workbench *workbench);

/**
 * Return nullptr if failed.
 * Note: remember call free on return value
 */
TENSOR_STACK_C_API ts_Program *ts_Workbench_compile(ts_Workbench *workbench, const ts_Module *module);

/**
 * Return false if failed.
 */
TENSOR_STACK_C_API ts_bool ts_Workbench_setup_device(ts_Workbench *workbench);

/**
 * Return false if failed
 */
TENSOR_STACK_C_API ts_bool ts_Workbench_setup_runtime(ts_Workbench *workbench);

TENSOR_STACK_C_API int32_t ts_Workbench_input_count(ts_Workbench *workbench);

TENSOR_STACK_C_API int32_t ts_Workbench_output_count(ts_Workbench *workbench);

TENSOR_STACK_C_API ts_bool ts_Workbench_run_hook(ts_Workbench *workbench, const char **node_names, int32_t len);

/**
 * Return NULL if failed.
 * Option can have:
 * 1. "--winograd" using winograd conv2d
 */
TENSOR_STACK_C_API ts_Workbench *ts_Workbench_Load_v2(const ts_Module *module, const ts_Device *device,
        const char *options);

/**
 * Return nullptr if failed.
 * Note: remember call free on return value
 * Option can have:
 * 1. "--winograd" using winograd conv2d
 */
TENSOR_STACK_C_API ts_Program *ts_Workbench_compile_v2(ts_Workbench *workbench, const ts_Module *module,
        const char *options);


#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_WORKBENCH_H
