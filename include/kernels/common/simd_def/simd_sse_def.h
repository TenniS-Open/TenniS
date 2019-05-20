#ifndef TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_SSE_DEF_H
#define TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_SSE_DEF_H

#ifdef TS_USE_SSE

#include <immintrin.h>

typedef struct __m128x2
{
    __m128 val[2];
} __m128x2;

using _simd_f32x4 = __m128;
using _simd_f32x4x2 = __m128x2;
using _simd_f32 = float;

inline _simd_f32x4 _simd_f32x4_load(const _simd_f32 *p) {
    return _mm_loadu_ps(p);
}

inline _simd_f32x4 _simd_f32x4_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d) {
    return _mm_set_ps(d, c, b, a);
}

inline void _simd_f32x4_store(_simd_f32 *p, _simd_f32x4 m) {
    _mm_storeu_ps(p, m);
}

inline _simd_f32x4 _simd_f32x4_add(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return _mm_add_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_sub(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return _mm_sub_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_mul(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return _mm_mul_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_div(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return _mm_div_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_max(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return _mm_max_ps(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_min(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return _mm_min_ps(lhs, rhs);
}

inline void _simd_f32x4_transpose4x4(_simd_f32x4& q0, _simd_f32x4& q1, _simd_f32x4& q2, _simd_f32x4& q3) {
    _MM_TRANSPOSE4_PS(q0, q1, q2, q3);
}

inline _simd_f32x4 _simd_f32x4_fmadd(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2) {
    return _mm_fmadd_ps(q0, q1, q2);
}


inline _simd_f32x4x2 _simd_f32x4x2_load(const _simd_f32 *p) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_loadu_ps(p);
    res.val[1] = _mm_loadu_ps(p + 4);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d,
                                     _simd_f32 e, _simd_f32 f, _simd_f32 g, _simd_f32 h) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_set_ps(d,c,b,a);
    res.val[1] = _mm_set_ps(h,g,f,e);
    return res;
}

inline void _simd_f32x4x2_store(_simd_f32 *p, _simd_f32x4x2 m) {
    _mm_storeu_ps(p, m.val[0]);
    _mm_storeu_ps(p + 4, m.val[1]);
}

inline _simd_f32x4x2 _simd_f32x4x2_add(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_add_ps(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_add_ps(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_sub(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_sub_ps(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_sub_ps(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_mul(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_mul_ps(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_mul_ps(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_div(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_div_ps(lhs.val[0], rhs.val[0]);
    res.val[1] = _mm_div_ps(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_fmadd(_simd_f32x4x2 q0, _simd_f32x4x2 q1, _simd_f32x4x2 q2) {
    _simd_f32x4x2 res;
    res.val[0] = _mm_fmadd_ps(q0.val[0], q1.val[0], q2.val[0]);
    res.val[1] = _mm_fmadd_ps(q0.val[1], q1.val[1], q2.val[1]);
    return res;
}
#endif //TS_USE_SSE

#endif //TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_SSE_DEF_H