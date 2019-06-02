//
// Created by keir on 2019/3/16.
//

#ifndef TENSORSTACK_API_COMMON_H
#define TENSORSTACK_API_COMMON_H

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

typedef int32_t ts_bool;

/**
 * Get last error message in current thread.
 * @return last error message.
 * @note Return NULL if failed.
 */
TENSOR_STACK_C_API const char *ts_last_error_message();

/**
 * Set error message in current thread.
 * @param message error message.
 */
TENSOR_STACK_C_API void ts_set_error_message(const char *message);

#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_API_COMMON_H
