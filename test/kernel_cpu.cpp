//
// Created by seeta on 2018/7/19.
//

#include <kernels/cpu/math_cpu.h>
#include <OpenBLAS/cblas.h>

int main() {
    using namespace ts;

    int i = 0;
    double A[6] = { 1.0, 2.0, 1.0, -3.0, 4.0, -1.0 };
    double B[6] = { 1.0, 2.0, 1.0, -3.0, 4.0, -1.0 };
    double C[9] = { .5, .5, .5, .5, .5, .5, .5, .5, .5 };
    double D[9] = { .5, .5, .5, .5, .5, .5, .5, .5, .5 };
    double E[9] = { .5, .5, .5, .5, .5, .5, .5, .5, .5 };
    double F[9] = { .5, .5, .5, .5, .5, .5, .5, .5, .5 };

    ts::cpu::math<double>::gemm(blas::ColMajor, blas::NoTrans, blas::NoTrans, 3, 3, 2, 1, A, 3, B, 2, 1, C, 3);

    return 0;
}