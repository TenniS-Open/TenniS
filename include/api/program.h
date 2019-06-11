//
// Created by kier on 19-5-7.
//

#ifndef TENSORSTACK_API_PROGORAM_H
#define TENSORSTACK_API_PROGORAM_H

#include "device.h"
#include "common.h"
#include "tensor.h"
#include "module.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ts_Program;
typedef struct ts_Program ts_Program;

// Workbench API

/**
 * Return NULL if failed.
 * Note: call ts_Workbench_setup_context, before compile, or call ts_Workbench_compile instead
 */
TENSOR_STACK_C_API ts_Program *ts_Program_Compile(const ts_Module *module, const ts_Device *device);

/**
 * Happen nothing if failed.
 */
TENSOR_STACK_C_API void ts_free_Program(const ts_Program *program);

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Program *ts_Program_clone(ts_Program *program);

TENSOR_STACK_C_API int32_t ts_Program_input_count(ts_Program *program);

TENSOR_STACK_C_API int32_t ts_Program_output_count(ts_Program *program);

/**
 * Return NULL if failed.
 * Note: call ts_Workbench_setup_context, before compile, or call ts_Workbench_compile instead
 * Option can have:
 * 1. "--float16" using float16 operator
 * 2. "--winograd" using winograd conv2d
 */
TENSOR_STACK_C_API ts_Program *ts_Program_Compile_v2(const ts_Module *module, const ts_Device *device,
        const char *options);


#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_PROGORAM_H
