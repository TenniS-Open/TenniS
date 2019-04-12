//
// Created by kier on 2018/12/22.
//

#ifndef TENSORSTACK_KERNELS_COMMON_SIMD_DEF_H
#define TENSORSTACK_KERNELS_COMMON_SIMD_DEF_H

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>

using _simd_f32x4 = float32x4_t;
using _simd_f32x4_x2 = float32x4x2_t;
using _simd_f32x2 = float32x2_t;
using _simd_f32 = float;

inline _simd_f32x4 _simd_f32x4_load(const _simd_f32 *p){
    return vld1q_f32(p);
}

inline _simd_f32x4 _simd_f32x4_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d){
    _simd_f32 array[4] = {a, b, c, d};
    return vld1q_f32(array);
}

inline void _simd_f32x4_store(_simd_f32 *p, _simd_f32x4 m){
    vst1q_f32(p, m);
}

inline _simd_f32x4 _simd_f32x4_add(_simd_f32x4 lhs, _simd_f32x4 rhs){
    return vaddq_f32(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_sub(_simd_f32x4 lhs, _simd_f32x4 rhs){
    return vsubq_f32(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_mul(_simd_f32x4 lhs, _simd_f32x4 rhs){
    return vmulq_f32(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_div(_simd_f32x4 lhs, _simd_f32x4 rhs){
    _simd_f32x4 recip = vrecpeq_f32(rhs);
    return vmulq_f32(lhs, recip);
}

inline _simd_f32x4 _simd_f32x4_max(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return vmaxq_f32(lhs, rhs);
}

inline _simd_f32x4 _simd_f32x4_min(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return vminq_f32(lhs, rhs);
}

inline void _simd_f32x4_transpose4x4(_simd_f32x4& q0, _simd_f32x4& q1, _simd_f32x4& q2, _simd_f32x4& q3) {

    _simd_f32x4_x2 q01 = vtrnq_f32(q0, q1);
    _simd_f32x4_x2 q23 = vtrnq_f32(q2, q3);

    _simd_f32x4 qq0 = q01.val[0];
    _simd_f32x2 d00 = vget_low_f32(qq0);
    _simd_f32x2 d01 = vget_high_f32(qq0);

    _simd_f32x4 qq1 = q01.val[1];
    _simd_f32x2 d10 = vget_low_f32(qq1);
    _simd_f32x2 d11 = vget_high_f32(qq1);

    _simd_f32x4 qq2 = q23.val[0];
    _simd_f32x2 d20 = vget_low_f32(qq2);
    _simd_f32x2 d21 = vget_high_f32(qq2);

    _simd_f32x4 qq3 = q23.val[1];
    _simd_f32x2 d30 = vget_low_f32(qq3);
    _simd_f32x2 d31 = vget_high_f32(qq3);

    q0 = vcombine_f32(d00, d20);
    q1 = vcombine_f32(d10, d30);
    q2 = vcombine_f32(d01, d21);
    q3 = vcombine_f32(d11, d31);
}

#elif TS_USE_SSE
#include <immintrin.h>

using _simd_f32x4 = __m128;
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

#else

#include <array>

using _simd_f32 = float;
using _simd_f32x4 = std::array<_simd_f32, 4>;

inline _simd_f32x4 _simd_f32x4_load(const _simd_f32 *p) {
    return { p[0], p[1], p[2], p[3] };
}

inline _simd_f32x4 _simd_f32x4_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d) {
    return { a, b, c, d };
}

inline void _simd_f32x4_store(_simd_f32 *p, _simd_f32x4 m) {
    p[0] = m[0];
    p[1] = m[1];
    p[2] = m[2];
    p[3] = m[3];
}

inline _simd_f32x4 _simd_f32x4_add(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return { lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3] };
}

inline _simd_f32x4 _simd_f32x4_sub(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return { lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3] };
}

inline _simd_f32x4 _simd_f32x4_mul(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return { lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2], lhs[3] * rhs[3] };
}

inline _simd_f32x4 _simd_f32x4_div(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return { lhs[0] / rhs[0], lhs[1] / rhs[1], lhs[2] / rhs[2], lhs[3] / rhs[3] };
}

inline _simd_f32x4 _simd_f32x4_max(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return { std::max(lhs[0],rhs[0]), std::max(lhs[1],rhs[1]), std::max(lhs[2],rhs[2]), std::max(lhs[3],rhs[3]) }
}

inline _simd_f32x4 _simd_f32x4_min(_simd_f32x4 lhs, _simd_f32x4 rhs) {
    return{ std::min(lhs[0],rhs[0]), std::min(lhs[1],rhs[1]), std::min(lhs[2],rhs[2]), std::min(lhs[3],rhs[3]) }
}

#endif



#endif //TENSORSTACK_KERNELS_COMMON_SIMD_DEF_H
