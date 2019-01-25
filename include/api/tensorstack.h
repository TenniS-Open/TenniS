//
// Created by kier on 2019/1/24.
//

#ifndef TENSORSTACK_API_TENSORSTACK_H
#define TENSORSTACK_API_TENSORSTACK_H

#define TS_EXTERN_C extern "C"

#if defined(_MSC_VER)
#   define TS_DLL_IMPORT __declspec(dllimport)
#   define TS_DLL_EXPORT __declspec(dllexport)
#   define TS_DLL_LOCAL
#else
#   if defined(__GNUC__) && __GNUC__ >= 4
#       define TS_DLL_IMPORT __attribute__((visibility("default")))
#       define TS_DLL_EXPORT __attribute__((visibility("default")))
#       define TS_DLL_LOCAL  __attribute__((visibility("hidden")))
#   else
#       define TS_DLL_IMPORT
#       define TS_DLL_EXPORT
#       define TS_DLL_LOCAL
#   endif
#endif

#if defined(BUILDING_TENSORSTACK)
#define TS_API TS_DLL_EXPORT
#else
#define TS_API TS_DLL_IMPORT
#endif

#ifdef __cplusplus
#   define TS_C_API TS_EXTERN_C TS_API
#else
#   define TS_C_API TS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

struct ts_Tensor;
struct ts_Module;
struct ts_Workbench;

enum ts_DTYPE {
    TS_VOID        = 0,
    TS_INT8        = 1,
    TS_UINT8       = 2,
    TS_INT16       = 3,
    TS_UINT16      = 4,
    TS_INT32       = 5,
    TS_UINT32      = 6,
    TS_INT64       = 7,
    TS_UINT64      = 8,
    TS_FLOAT16     = 9,
    TS_FLOAT32     = 10,
    TS_FLOAT64     = 11,
    TS_PTR         = 12,              ///< for ptr type, with length of sizeof(void*) bytes
    TS_CHAR8       = 13,            ///< for char saving string
    TS_CHAR16      = 14,           ///< for char saving utf-16 string
    TS_CHAR32      = 15,           ///< for char saving utf-32 string
    TS_UNKNOWN8    = 16,        ///< for self define type, with length of 1 byte
    TS_UNKNOWN16   = 17,
    TS_UNKNOWN32   = 18,
    TS_UNKNOWN64   = 19,
    TS_UNKNOWN128  = 20,

    TS_BOOLEAN     = 21,    // bool type, using byte in native
    TS_COMPLEX32   = 22,  // complex 32(16 + 16)
    TS_COMPLEX64   = 23,  // complex 64(32 + 32)
    TS_COMPLEX128  = 24,  // complex 128(64 + 64)

    TS_SINK8Q0     = 25,
    TS_SINK8Q1     = 26,
    TS_SINK8Q2     = 27,
    TS_SINK8Q3     = 28,
    TS_SINK8Q4     = 29,
    TS_SINK8Q5     = 30,
    TS_SINK8Q6     = 31,
    TS_SINK8Q7     = 32,
};

struct ts_Device {
    const char *type;
    int32_t id;
};

/**
 * Happen nothing if failed.
 */
TS_C_API void ts_setup();

/**
 * Return NULL if failed.
 */
TS_C_API const char *ts_last_error_message();

// Tensor's API

/**
 * Return NULL if failed.
 */
TS_C_API ts_Tensor *ts_new_Tensor(int32_t *shape, int32_t shape_len, ts_DTYPE dtype, const void *data);

/**
 * Happen nothing if failed.
 */
TS_C_API void ts_free_Tensor(const ts_Tensor *tensor);

/**
 * Return non-zero if failed.
 */
TS_C_API int32_t ts_Tensor_shape(ts_Tensor *tensor, int32_t **shape, int32_t *shape_len);

/**
 * Return TS_VOID if failed.
 */
TS_C_API ts_DTYPE ts_Tensor_dtype(ts_Tensor *tensor);

/**
 * Return NULL if failed.
 */
TS_C_API void* ts_Tensor_data(ts_Tensor *tensor);

/**
 * Return NULL if failed.
 */
TS_C_API ts_Tensor *ts_Tensor_clone(ts_Tensor *tensor);

// Module's API

enum ts_SerializationFormat {
    TS_BINARY   = 0,    // BINARY file format
    TS_TEXT     = 1,    // TEXT file format
};

/**
 * Return NULL if failed.
 */
TS_C_API ts_Module *ts_Module_Load(const char *filename, ts_SerializationFormat format);

/**
 * Happen nothing if failed.
 */
TS_C_API void ts_free_Module(const ts_Module *module);

// Workbench API

/**
 * Return NULL if failed.
 */
TS_C_API ts_Workbench *ts_Workbench_Load(ts_Module *module, const ts_Device *device);

/**
 * Happen nothing if failed.
 */
TS_C_API void ts_free_Workbench(const ts_Workbench *workbench);

/**
 * Return NULL if failed.
 */
TS_C_API ts_Workbench *ts_Workbench_clone(ts_Workbench *workbench);

/**
 * Return non-zero if failed.
 */
TS_C_API int32_t ts_Workbench_input(ts_Workbench *workbench, int32_t i, const ts_Tensor *tensor);

/**
 * Return non-zero if failed.
 */
TS_C_API int32_t ts_Workbench_input_by_name(ts_Workbench *workbench, const char *name, const ts_Tensor *tensor);

/**
 * Return non-zero if failed.
 */
TS_C_API int32_t ts_Workbench_run(ts_Workbench *workbench);

/**
 * Return NULL if failed.
 */
TS_C_API int32_t ts_Workbench_output(ts_Workbench *workbench, int32_t i, ts_Tensor *tensor);

/**
 * Return NULL if failed.
 */
TS_C_API int32_t ts_Workbench_output_by_name(ts_Workbench *workbench, const char *name, ts_Tensor *tensor);

/**
 * Happen Nothing if failed.
 */
TS_C_API void ts_Workbench_set_computing_thread_number(ts_Workbench *workbench, int32_t number);

#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_TENSORSTACK_H
