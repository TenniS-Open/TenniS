//
// Created by kier on 2018/12/21.
//

#ifndef TENSORSTACK_KERNELS_COMMON_SIMD_H
#define TENSORSTACK_KERNELS_COMMON_SIMD_H

#include "simd_def.h"

namespace ts {
    template<typename T, int M>
    class simd_base {
    public:
        using self = simd_base;
        using base = T;
        static const int width = M;
    };

    template<typename T, int M>
    class simd : public simd_base<T, M> {
    public:
        using self = simd;
        using supper = simd_base<T, M>;

        void store(typename supper::base *p) const;
    };

    using float32x4 = simd<float, 4>;

    template<typename T, int M>
    inline T sum(const simd<T, M> &value) {
        T a[M];
        value.store(a);
        T sum = 0;
        for (int i = 0; i < M; ++i) sum += a[i];
        return sum;
    }

    template<typename T>
    inline T sum(const simd<T, 4> &value) {
        T a[4];
        value.store(a);
        return a[0] + a[1] + a[2] + a[3];
    }

    template<typename T, int M>
    inline const simd<T, M> &operator+=(simd<T, M> &lhs, const simd<T, M> &rhs) {
        return lhs = lhs + rhs;
    }

    template<>
    class simd<float, 4> : public simd_base<float, 4> {
    public:
        using self = simd;
        using type = _simd_f32x4;

        type value;

        simd() = default;

        simd(type value) : value(value) {}

        simd(base a) : simd(a, a, a, a) {}

        simd(int a) : simd(base(a)) {}

        simd(const base *p) : value(_simd_f32x4_load(p)) {}

        simd(base a, base b, base c, base d) : value(_simd_f32x4_set(a, b, c, d)) {}

        void store(base *p) const { _simd_f32x4_store(p, value); }
    };

    inline simd<float, 4> operator+(const simd<float, 4> &lhs, const simd<float, 4> &rhs) {
        return _simd_f32x4_add(lhs.value, rhs.value);
    }

    inline simd<float, 4> operator-(const simd<float, 4> &lhs, const simd<float, 4> &rhs) {
        return _simd_f32x4_sub(lhs.value, rhs.value);
    }

    inline simd<float, 4> operator*(const simd<float, 4> &lhs, const simd<float, 4> &rhs) {
        return _simd_f32x4_mul(lhs.value, rhs.value);
    }

    inline simd<float, 4> operator/(const simd<float, 4> &lhs, const simd<float, 4> &rhs) {
        return _simd_f32x4_div(lhs.value, rhs.value);
    }

}

#endif //TENSORSTACK_KERNELS_COMMON_SIMD_H
