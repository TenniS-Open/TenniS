//
// Created by kier on 2019/1/24.
//

#ifndef TENSORSTACK_API_TENSORSTACK_H
#define TENSORSTACK_API_TENSORSTACK_H

#define TENSOR_STACK_EXTERN_C extern "C"

#if defined(_MSC_VER)
#   define TENSOR_STACK_DLL_IMPORT __declspec(dllimport)
#   define TENSOR_STACK_DLL_EXPORT __declspec(dllexport)
#   define TENSOR_STACK_DLL_LOCAL
#else
#   if defined(__GNUC__) && __GNUC__ >= 4
#       define TENSOR_STACK_DLL_IMPORT __attribute__((visibility("default")))
#       define TENSOR_STACK_DLL_EXPORT __attribute__((visibility("default")))
#       define TENSOR_STACK_DLL_LOCAL  __attribute__((visibility("hidden")))
#   else
#       define TENSOR_STACK_DLL_IMPORT
#       define TENSOR_STACK_DLL_EXPORT
#       define TENSOR_STACK_DLL_LOCAL
#   endif
#endif

#if defined(BUILDING_TENSORSTACK)
#define TENSOR_STACK_API TENSOR_STACK_DLL_EXPORT
#else
#define TENSOR_STACK_API TENSOR_STACK_DLL_IMPORT
#endif

#ifdef __cplusplus
#   define TENSOR_STACK_C_API TENSOR_STACK_EXTERN_C TENSOR_STACK_API
#else
#   define TENSOR_STACK_C_API TENSOR_STACK_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

struct ts_Tensor;
typedef struct ts_Tensor ts_Tensor;

struct ts_Module;
typedef struct ts_Module ts_Module;

struct ts_Workbench;
typedef struct ts_Workbench ts_Workbench;

struct ts_ImageFilter;
typedef struct ts_ImageFilter ts_ImageFilter;

typedef int32_t ts_bool;

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
    TS_FLOAT32     = 10,
    TS_FLOAT64     = 11,
};
typedef enum ts_DTYPE ts_DTYPE;

struct ts_Device {
    const char *type;
    int32_t id;
};
typedef struct ts_Device ts_Device;

/**
 * Happen nothing if failed.
 */
TENSOR_STACK_C_API ts_bool ts_setup();

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API const char *ts_last_error_message();

// Tensor's API

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Tensor *ts_new_Tensor(int32_t *shape, int32_t shape_len, ts_DTYPE dtype, const void *data);

/**
 * Happen nothing if failed.
 */
TENSOR_STACK_C_API void ts_free_Tensor(const ts_Tensor *tensor);

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API const int32_t *ts_Tensor_shape(ts_Tensor *tensor);

/**
 * Return zero if failed.
 */
TENSOR_STACK_C_API int32_t ts_Tensor_shape_size(ts_Tensor *tensor);

/**
 * Return TS_VOID if failed.
 */
TENSOR_STACK_C_API ts_DTYPE ts_Tensor_dtype(ts_Tensor *tensor);

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API void* ts_Tensor_data(ts_Tensor *tensor);

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Tensor *ts_Tensor_clone(ts_Tensor *tensor);

/**
 * Return false if failed.
 */
TENSOR_STACK_C_API ts_bool ts_Tensor_sync_cpu(ts_Tensor *tensor);

// Module's API

enum ts_SerializationFormat {
    TS_BINARY   = 0,    // BINARY file format
    TS_TEXT     = 1,    // TEXT file format
};

/**
 * Return NULL if failed.
 */
TENSOR_STACK_C_API ts_Module *ts_Module_Load(const char *filename, ts_SerializationFormat format);

/**
 * Happen nothing if failed.
 */
TENSOR_STACK_C_API void ts_free_Module(const ts_Module *module);

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

/**
 * Image filter API
 */


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

TENSOR_STACK_C_API ts_bool ts_ImageFilter_resize(ts_ImageFilter *filter, int width, int height);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_center_crop(ts_ImageFilter *filter, int width, int height);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_channel_swap(ts_ImageFilter *filter, const int *shuffle, int32_t len);

TENSOR_STACK_C_API ts_bool ts_ImageFilter_to_chw(ts_ImageFilter *filter);

TENSOR_STACK_C_API ts_bool ts_Workbench_bind_filter(ts_Workbench *workbench, int32_t i, ts_ImageFilter *filter);

TENSOR_STACK_C_API ts_bool ts_Workbench_bind_filter_by_name(ts_Workbench *workbench, const char *name, ts_ImageFilter *filter);


#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_TENSORSTACK_H
