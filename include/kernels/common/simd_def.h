//
// Created by kier on 2018/12/22.
//

#ifndef TENSORSTACK_KERNELS_COMMON_SIMD_DEF_H
#define TENSORSTACK_KERNELS_COMMON_SIMD_DEF_H

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>

using _simd_f32x4 = float32x4_t;
using _simd_f32x4x2 = float32x4x2_t;
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

    /*
    * q0 = (s00,s01,s02,s03)
    * q1 = (s10,s11,s12,s13)
    * q2 = (s20,s21,s22,s23)
    * q3 = (s30,s31,s32,s33)
    */
    /*
    * q01 = (s00,s10,s02,s12),(s01,s11,s03,s13)
    * q02 = (s20,s30,s22,s32),(s21,s31,s23,s33)
    */
    _simd_f32x4x2 q01 = vtrnq_f32(q0, q1);
    _simd_f32x4x2 q23 = vtrnq_f32(q2, q3);

    _simd_f32x2 d00 = vget_low_f32(q01.val[0]);
    _simd_f32x2 d01 = vget_high_f32(q01.val[0]);

    _simd_f32x2 d10 = vget_low_f32(q01.val[1]);
    _simd_f32x2 d11 = vget_high_f32(q01.val[1]);

    _simd_f32x2 d20 = vget_low_f32(q23.val[0]);
    _simd_f32x2 d21 = vget_high_f32(q23.val[0]);

    _simd_f32x2 d30 = vget_low_f32(q23.val[1]);
    _simd_f32x2 d31 = vget_high_f32(q23.val[1]);

    q0 = vcombine_f32(d00, d20);
    q1 = vcombine_f32(d10, d30);
    q2 = vcombine_f32(d01, d21);
    q3 = vcombine_f32(d11, d31);
}

inline _simd_f32x4 _simd_f32x4_fmadd(const _simd_f32x4& q0, const _simd_f32x4& q1, const _simd_f32x4& q2) {
    return vmlaq_f32(q2, q0, q1);
    //_simd_f32x4 mul_tmp = vmulq_f32(q0, q1);
    //return vaddq_f32(mul_tmp, q2);

}


inline _simd_f32x4x2 _simd_f32x4x2_load(const _simd_f32 *p) {
    _simd_f32x4x2 res;
    res.val[0] = vld1q_f32(p); 
    res.val[1] = vld1q_f32(p + 4);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_set(_simd_f32 a, _simd_f32 b, _simd_f32 c, _simd_f32 d,
    _simd_f32 e, _simd_f32 f, _simd_f32 g, _simd_f32 h) {
    _simd_f32x4x2 res;
    _simd_f32 array_0[4] = { a, b, c, d };
    _simd_f32 array_1[4] = { e, f, g, h };
    res.val[0] = vld1q_f32(array_0); res.val[1] = vld1q_f32(array_1);
    return res;
}

inline void _simd_f32x4x2_store(_simd_f32 *p, _simd_f32x4x2 m) {
    vst1q_f32(p, m.val[0]);
    vst1q_f32(p + 4, m.val[1]);
}

inline _simd_f32x4x2 _simd_f32x4x2_add(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    res.val[0] = vaddq_f32(lhs.val[0], rhs.val[0]);
    res.val[1] = vaddq_f32(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_sub(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    res.val[0] = vsubq_f32(lhs.val[0], rhs.val[0]);
    res.val[1] = vsubq_f32(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_mul(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    res.val[0] = vmulq_f32(lhs.val[0], rhs.val[0]);
    res.val[1] = vmulq_f32(lhs.val[1], rhs.val[1]);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_div(_simd_f32x4x2 lhs, _simd_f32x4x2 rhs) {
    _simd_f32x4x2 res;
    _simd_f32x4 recip_0 = vrecpeq_f32(rhs.val[0]);
    _simd_f32x4 recip_1 = vrecpeq_f32(rhs.val[1]);
    res.val[0] = vmulq_f32(lhs.val[0], recip_0);
    res.val[1] = vmulq_f32(lhs.val[1], recip_1);
    return res;
}

inline _simd_f32x4x2 _simd_f32x4x2_fmadd(_simd_f32x4x2 q0, _simd_f32x4x2 q1, _simd_f32x4x2 q2) {
    _simd_f32x4x2 res;
    res.val[0] = vmlaq_f32(q2.val[0], q0.val[0], q1.val[0]);
    res.val[1] = vmlaq_f32(q2.val[1], q0.val[1], q1.val[1]);
    //_simd_f32x4 mul_tmp_0 = vmulq_f32(q0.val[0], q1.val[0]);
    //_simd_f32x4 mul_tmp_1 = vmulq_f32(q0.val[1], q1.val[1]);
    //res.val[0] = vaddq_f32(mul_tmp_0, q2.val[0]);
    //res.val[1] = vaddq_f32(mul_tmp_1, q2.val[1]);
    return res;
}

#elif TS_USE_SSE
#include <immintrin.h>

using _simd_f32x4 = __m128;
using _simd_f32x4x2 = __m256;
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
