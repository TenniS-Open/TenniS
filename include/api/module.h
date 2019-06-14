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

/**
 * Module contains struct and input, output of networks.
 */
struct ts_Module;
typedef struct ts_Module ts_Module;

/**
 * Serialized module format
 */
enum ts_SerializationFormat {
    TS_BINARY   = 0,    // BINARY file format
    TS_TEXT     = 1,    // TEXT file format
};
typedef enum ts_SerializationFormat ts_SerializationFormat;

// Module's API

/**
 * Load module from given filename.
 * @param filename
 * @param format @sa ts_SerializationFormat, only support TS_BINARY in this version.
 * @return New reference. Return NULL if failed.
 * @note call @see ts_free_Module to free ts_Module
 */
TENSOR_STACK_C_API ts_Module *ts_Module_Load(const char *filename, ts_SerializationFormat format);

/**
 * Load module from given stream.
 * @param obj object pointer pass to reader
 * @param reader stream reader
 * @param format @sa ts_SerializationFormat, only support TS_BINARY in this version.
 * @return New reference. Return NULL if failed.
 * @note call @see ts_free_Module to free ts_Module
 */
TENSOR_STACK_C_API ts_Module *ts_Module_LoadFromStream(void *obj, ts_stream_read *reader, ts_SerializationFormat format);

/**
 * Free module.
 * @param module the return value of ts_Module_Load<XXX>
 * Happen nothing if failed.
 */
TENSOR_STACK_C_API void ts_free_Module(const ts_Module *module);


#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_MODULE_H
