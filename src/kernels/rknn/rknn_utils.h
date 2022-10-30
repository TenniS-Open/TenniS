//
// Created by yang on 2019/11/13.
//

#ifndef TENSORSTACK_KERNELS_RKNN_UTILS_H
#define TENSORSTACK_KERNELS_RKNN_UTILS_H

#include <string>

#include "core/dtype.h"

#include "dll/rknn_dll.h"

namespace ts {
//    using rknn_context = int;
    static inline std::string get_rknn_error_str(int error_code) {
        std::string str = "";
        switch (error_code) {
            case 0 :
                str = "execute success";
                break;
            case -1 :
                str = "execute failed";
                break;
            case -2 :
                str = "execute timeout";
                break;
            case -3 :
                str = "device is unavailable";
                break;
            case -4 :
                str = "memory malloc fail";
                break;
            case -5 :
                str = "parameter is invalid";
                break;
            case -6 :
                str = "model is invalid";
                break;
            case -7 :
                str = "context is invalid";
                break;
            case -8 :
                str = "input is invalid";
                break;
            case -9 :
                str = "output is invalid";
                break;
            case -10 :
                str = "the device is unmatch, please update rknn sdk and npu driver/firmware";
                break;
            case -11 :
                str = "This RKNN model use pre_compile mode, but not compatible with current driver.";
                break;
            case -12 :
                str = "This RKNN model set optimization level, but not compatible with current driver.";
                break;
            case -13 :
                str = "This RKNN model set target platform, but not compatible with current platform.";
                break;
            default :
                break;
        }
        return str;
    }

    static inline DTYPE get_ts_type_from_rknn(rknn_tensor_type rknn_type) {
        DTYPE type = VOID;
        switch (rknn_type) {
            case RKNN_TENSOR_FLOAT32 :
                type = FLOAT32;
                break;
            case RKNN_TENSOR_FLOAT16 :
                type = FLOAT16;
                break;
            case RKNN_TENSOR_INT8 :
                type = INT8;
                break;
            case RKNN_TENSOR_UINT8 :
                type = UINT8;
                break;
            case RKNN_TENSOR_INT16 :
                type = INT16;
                break;
            case RKNN_TENSOR_UINT16 :
                type = UINT16;
                break;
            case RKNN_TENSOR_INT32 :
                type = INT32;
                break;
            case RKNN_TENSOR_UINT32 :
                type = UINT32;
                break;
            case RKNN_TENSOR_INT64 :
                type = INT64;
                break;
            case RKNN_TENSOR_BOOL :
                type = BOOLEAN;
                break;
            default:
                break;
        }
        return type;
    }

    static inline rknn_tensor_type get_rknn_type(DTYPE ts_type) {
        rknn_tensor_type type = RKNN_TENSOR_FLOAT32;
        switch (ts_type) {
            case FLOAT32 :
                type = RKNN_TENSOR_FLOAT32;
                break;
            case FLOAT16 :
                type = RKNN_TENSOR_FLOAT16;
                break;
            case INT8 :
                type = RKNN_TENSOR_INT8;
                break;
            case UINT8 :
                type = RKNN_TENSOR_UINT8;
                break;
            case INT16 :
                type = RKNN_TENSOR_INT16;
                break;
            case UINT16 :
                type = RKNN_TENSOR_UINT16;
                break;
            case INT32 :
                type = RKNN_TENSOR_INT32;
                break;
            case UINT32 :
                type = RKNN_TENSOR_UINT32;
                break;
            case INT64 :
                type = RKNN_TENSOR_INT64;
                break;
            case BOOLEAN :
                type = RKNN_TENSOR_BOOL;
                break;
            default:
                break;
        }
        return type;
    }

    static inline const char *get_rknn_format_string(int fmt) {
        switch (fmt) {
            case RKNN_TENSOR_NCHW:
                return "NCHW";
            case RKNN_TENSOR_NHWC:
                return "NHWC";
            case RKNN_TENSOR_NC1HWC2:
                return "NC1HWC2";
            case RKNN_TENSOR_UNDEFINED:
                return "UNDEFINED";
            default:
                break;
        }
        return "unknown";
    }

} //ts

#endif //TENSORSTACK_KERNELS_RKNN_UTILS_H
