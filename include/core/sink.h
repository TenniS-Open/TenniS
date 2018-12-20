//
// Created by kier on 2018/12/19.
//

#ifndef TENSORSTACK_CORE_SINK_H
#define TENSORSTACK_CORE_SINK_H

#include <cstdint>

namespace ts {
    using sink8_t = uint8_t;

    extern const int code_map[];

    union __union_float32_int32 {
        __union_float32_int32() = default;
        __union_float32_int32(float f) : f(f) {}
        __union_float32_int32(uint32_t i) : i(i) {}
        __union_float32_int32(int32_t i) : i(uint32_t(i)) {}
        float f;
        uint32_t i;
    };

    inline float to_float(sink8_t s, int Q) {
        // static const int float_width = 31;
        // static const int sink8_tail_width = 7;
        // static const int float_tail_width = 23;

        // TODO: Check if it's little endian
        int32_t s_tail = s & 0x7f;
        int32_t sign = (s & 0x80) << (31 - 7);
        if (s_tail == 0) return __union_float32_int32(sign).f;
        auto first_one = code_map[s_tail];
        int32_t exp = first_one - Q - 1;
        int32_t tail = s_tail << (23 + 1 - first_one);

        __union_float32_int32 u = uint32_t(sign | ((exp + 127) << 23) | (tail & 0x007fffff));

        return u.f;
    }

    inline sink8_t to_sink(float f, int Q) {
        // static const int float_width = 31;
        // static const int sink8_tail_width = 7;
        // static const int float_tail_width = 23;

        // TODO: Check if it's little endian
        __union_float32_int32 u = f;

        uint32_t sign = (u.i & 0x80000000) >> (31 - 7);
        int32_t exp = int32_t((u.i & 0x7f800000) >> 23) - 127;
        uint32_t tail = (u.i & 0x007fffff) | 0x00800000;

        auto right_shift = (23 - exp - Q);
        uint32_t fixed = right_shift > 23 ? 0 : (right_shift <= 23 - 7 ? 0x7f : tail >> right_shift);

        auto s = sink8_t((sign | fixed) & 0xff);

        return s;
    }
}

#endif //TENSORSTACK_CORE_SINK_H
