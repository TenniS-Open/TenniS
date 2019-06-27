#ifndef TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_AVX_DEF_H
#define TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_AVX_DEF_H

#ifdef TS_USE_AVX
#include <immintrin.h>
#include <emmintrin.h>

using _simd_f32x4 = __m128;
using _simd_f32x4x2 = __m256;
using _simd_f32 = float;
using _simd_int32x4 = __m128i;
using _simd_int32 = int32_t;
using _simd_int32x4x2 = __m256i;

inline _simd_int32x4 _simd_int32x4_load(const _simd_int32* p) {
    return _mm_loadu_si128((_simd_int32x4*)p);
}

inline _simd_int32x4 _simd_int32x4_set(_simd_int32 a, _simd_int32 b, _simd_int32 c, _simd_int32 d) {
    return _mm_set_epi32(d, c, b, a);
}

inline void _simd_int32x4_store(_simd_int32 *p, _simd_int32x4 m) {
    _mm_store_si128((_simd_int32x4*)p, m);
}

inline _simd_int32x4 _simd_int32x4_add(_simd_int32x4 lhs, _simd_int32x4 rhs) {
    return _mm_add_epi32(lhs, rhs);
}

inline _simd_int32x4 _simd_int32x4_sub(_simd_int32x4 lhs, _simd_int32x4 rhs) {
    return _mm_sub_epi32(lhs, rhs);
}

inline _simd_int32x4 _simd_int32x4_mul(_simd_int32x4 lhs, _simd_int32x4 rhs) {
    return _mm_mul_epi32(lhs, rhs);
}

inline _simd_int32x4x2 _simd_int32x4x2_load(const _simd_int32* p) {
    return _mm256_loadu_si256((_simd_int32x4x2*)p);
}

inline _simd_int32x4x2 _simd_int32x4x2_set(_simd_int32 a, _simd_int32 b, _simd_int32 c, _simd_int32 d, 
                                           _simd_int32 e, _simd_int32 f, _simd_int32 g, _simd_int32 h) {
    return _mm256_set_epi32(h, g, f, e, d, c, b, a);
}

inline void _simd_int32x4x2_store(_simd_int32 *p, _simd_int32x4x2 m) {
    _mm256_store_si256((_simd_int32x4x2*)p, m);
}

inline _simd_int32x4x2 _simd_int32x4x2_add(_simd_int32x4x2 lhs, _simd_int32x4x2 rhs) {
    return _mm256_add_epi32(lhs, rhs);
}

inline _simd_int32x4x2 _simd_int32x4x2_sub(_simd_int32x4x2 lhs, _simd_int32x4x2 rhs) {
    return _mm256_sub_epi32(lhs, rhs);
}

inline _simd_int32x4x2 _simd_int32x4x2_mul(_simd_int32x4x2 lhs, _simd_int32x4x2 rhs) {
    return _mm256_mul_epi32(lhs, rhs);
}


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
    return _mm256_loadu_ps(p);
}

inline _simd_f32x4x2 _simd_f32x4x2_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d,
                                     _simd_f32 e, _simd_f32 f, _simd_f32 g, _simd_f32 h) {
    return _mm256_set_ps(h, g, f, e, d, c, b, a);
}

inline void _simd_f32x4x2_store(_simd_f32 *p, _simd_f32x4x2 m) {
    _mm256_storeu_ps(p, m);
}

inline _simd_f32x4x2 _simd_f32x4x2_add(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    return _mm256_add_ps(lhs, rhs);
}

inline _simd_f32x4x2 _simd_f32x4x2_sub(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    return _mm256_sub_ps(lhs, rhs);
}

inline _simd_f32x4x2 _simd_f32x4x2_mul(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    return _mm256_mul_ps(lhs, rhs);
}

inline _simd_f32x4x2 _simd_f32x4x2_div(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    return _mm256_div_ps(lhs, rhs);
}

inline _simd_f32x4x2 _simd_f32x4x2_fmadd(_simd_f32x4x2 q0, _simd_f32x4x2 q1, _simd_f32x4x2 q2) {
    return _mm256_fmadd_ps(q0, q1, q2);
}

//cast
inline _simd_int32x4x2 _simd_floatx4x2_to_int32x4x2(_simd_f32x4x2 src) {
    return _mm256_cvtps_epi32(src);
}

#endif //TS_USE_AVX

#endif //TENSORSTACK_KERNELS_COMMON_SIMD_DEF_SIMD_AVX_DEF_H