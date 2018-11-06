//
// Created by kier on 2018/5/25.
//

#ifndef TENSORSTACK_CORE_TYPE_H
#define TENSORSTACK_CORE_TYPE_H

#include <cstdint>

namespace ts {
    enum DTYPE {
        VOID,
        INT8,
        UINT8,
        INT16,
        UINT16,
        INT32,
        UINT32,
        INT64,
        UINT64,
        FLOAT16,
        FLOAT32,
        FLOAT64,
        PTR,              ///< for ptr type, with length of sizeof(void*) bytes
        CHAR8,            ///< for char saving string
        CHAR16,           ///< for char saving utf-16 string
        CHAR32,           ///< for char saving utf-32 string
        UNKNOWN8,        ///< for self define type, with length of 1 byte
        UNKNOWN16,
        UNKNOWN32,
        UNKNOWN64,
        UNKNOWN128,
    };

    inline int type_bytes(DTYPE dtype) {
        static const auto FakeUsagePtr = (void *) (0x19910929);
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
            case FLOAT16: return 2;
            case FLOAT32: return 4;
            case FLOAT64: return 6;
            case PTR: return sizeof(FakeUsagePtr);
            case CHAR8: return 1;
            case CHAR16: return 2;
            case CHAR32: return 4;
            case UNKNOWN8: return 1;
            case UNKNOWN16: return 2;
            case UNKNOWN32: return 4;
            case UNKNOWN64: return 8;
            case UNKNOWN128: return 16;
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
            case FLOAT16: return "float16";
            case FLOAT32: return "float32";
            case FLOAT64: return "float64";
            case PTR: return "pointer";
            case CHAR8: return "char8";
            case CHAR16: return "char16";
            case CHAR32: return "char32";
            case UNKNOWN8: return "unknown8";
            case UNKNOWN16: return "unknown16";
            case UNKNOWN32: return "unknown32";
            case UNKNOWN64: return "unknown64";
            case UNKNOWN128: return "unknown128";
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
    template <> struct dtype<FLOAT16> { using declare = uint16_t; }; ///< not specific doing method
    template <> struct dtype<FLOAT32> { using declare = float; };
    template <> struct dtype<FLOAT64> { using declare = double; };
    template <> struct dtype<PTR> { using declare = void*; };
    template <> struct dtype<CHAR8> { using declare = char; };
    template <> struct dtype<CHAR16> { using declare = char16_t; };
    template <> struct dtype<CHAR32> { using declare = char32_t; };

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
    template <> struct dtypeid<void*> { static const DTYPE id = PTR; };
    template <> struct dtypeid<char> { static const DTYPE id = CHAR8; };
    template <> struct dtypeid<char16_t> { static const DTYPE id = CHAR16; };
    template <> struct dtypeid<char32_t> { static const DTYPE id = CHAR32; };
}


#endif //TENSORSTACK_CORE_TYPE_H
