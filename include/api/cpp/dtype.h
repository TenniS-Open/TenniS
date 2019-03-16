//
// Created by kier on 2019/3/16.
//

#ifndef TENSORSTACK_API_CPP_DTYPE_H
#define TENSORSTACK_API_CPP_DTYPE_H

#include "../tensor.h"

namespace ts {
    namespace api {
        enum DTYPE {
            VOID = TS_VOID,
            INT8 = TS_INT8,
            UINT8 = TS_UINT8,
            INT16 = TS_INT16,
            UINT16 = TS_UINT16,
            INT32 = TS_INT32,
            UINT32 = TS_UINT32,
            INT64 = TS_INT64,
            UINT64 = TS_UINT64,
            FLOAT32 = TS_FLOAT32,
            FLOAT64 = TS_FLOAT64,
            CHAR8 = TS_CHAR8,
        };

        inline int type_bytes(DTYPE dtype) {
            switch (dtype) {
                case VOID: return 0;
                case INT8: return 1;
                case UINT8: return 1;
                case INT16: return 2;
                case UINT16: return 2;
                case INT32: return 4;
                case UINT32: return 4;
                case INT64: return 8;
                case UINT64: return 8;
                case FLOAT32: return 4;
                case FLOAT64: return 8;
                case CHAR8: return 1;
            }
            return 0;
        }

        inline const char *type_str(DTYPE dtype) {
            switch (dtype) {
                case VOID: return "void";
                case INT8: return "int8";
                case UINT8: return "uint8";
                case INT16: return "int64";
                case UINT16: return "uint64";
                case INT32: return "int32";
                case UINT32: return "uint32";
                case INT64: return "int64";
                case UINT64: return "uint64";
                case FLOAT32: return "float32";
                case FLOAT64: return "float64";
                case CHAR8: return "char8";
            }
            return "unknown";
        }

        template <DTYPE T> struct dtype { using declare = void; };

        template <> struct dtype<VOID> { using declare = void; };
        template <> struct dtype<INT8> { using declare = int8_t; };
        template <> struct dtype<UINT8> { using declare = uint8_t; };
        template <> struct dtype<INT16> { using declare = int16_t; };
        template <> struct dtype<UINT16> { using declare = uint16_t; };
        template <> struct dtype<INT32> { using declare = int32_t; };
        template <> struct dtype<UINT32> { using declare = uint32_t; };
        template <> struct dtype<INT64> { using declare = int64_t; };
        template <> struct dtype<UINT64> { using declare = uint64_t; };
        template <> struct dtype<FLOAT32> { using declare = float; };
        template <> struct dtype<FLOAT64> { using declare = double; };
        template <> struct dtype<CHAR8> { using declare = char; };

        template <typename T> struct dtypeid { static const DTYPE id = VOID; };

        template <> struct dtypeid<void> { static const DTYPE id = VOID; };
        template <> struct dtypeid<int8_t> { static const DTYPE id = INT8; };
        template <> struct dtypeid<uint8_t> { static const DTYPE id = UINT8; };
        template <> struct dtypeid<int16_t> { static const DTYPE id = INT16; };
        template <> struct dtypeid<uint16_t> { static const DTYPE id = UINT16; };
        template <> struct dtypeid<int32_t> { static const DTYPE id = INT32; };
        template <> struct dtypeid<uint32_t> { static const DTYPE id = UINT32; };
        template <> struct dtypeid<int64_t> { static const DTYPE id = INT64; };
        template <> struct dtypeid<uint64_t> { static const DTYPE id = UINT64; };
        template <> struct dtypeid<float> { static const DTYPE id = FLOAT32; };
        template <> struct dtypeid<double> { static const DTYPE id = FLOAT64; };
        template <> struct dtypeid<char> { static const DTYPE id = CHAR8; };
    }
}

#endif //TENSORSTACK_API_CPP_DTYPE_H
