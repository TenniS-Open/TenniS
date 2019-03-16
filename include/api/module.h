//
// Created by keir on 2019/3/16.
//

#ifndef TENSORSTACK_API_MODULE_H
#define TENSORSTACK_API_MODULE_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ts_Module;
typedef struct ts_Module ts_Module;

enum ts_SerializationFormat {
    TS_BINARY   = 0,    // BINARY file format
    TS_TEXT     = 1,    // TEXT file format
};

// Module's API

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Module *ts_Module_Load(const char *filename, ts_SerializationFormat format);

/**
 * Happen nothing if failed.
 */
TENSOR_STACK_C_API void ts_free_Module(const ts_Module *module);

#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_MODULE_H
