//
// Created by keir on 2019/3/16.
//

#ifndef TENSORSTACK_API_MODULE_H
#define TENSORSTACK_API_MODULE_H

#include "common.h"
#include "stream.h"
#include "device.h"

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
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Module *ts_Module_LoadFromStream(void *obj, ts_stream_read *reader, ts_SerializationFormat format);

/**
 * Happen nothing if failed.
 */
TENSOR_STACK_C_API void ts_free_Module(const ts_Module *module);

/**
 * Return new reference
 * Option can have:
 * 1. "--float16" using float16 operator
 */
TENSOR_STACK_C_API ts_Module *ts_Module_translate(const ts_Module *module, const ts_Device *device, const char *options);


#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_MODULE_H
