//
// Created by seeta on 2018/7/19.
//

#ifndef TENSORSTACK_KERNELS_COMMON_BLAS_H
#define TENSORSTACK_KERNELS_COMMON_BLAS_H

// #include <OpenBLAS/cblas.h>

namespace ts {
    namespace blas {
        enum Order {
            RowMajor = 101,
            ColMajor = 102
        };
        enum Transpose {
            NoTrans = 111,
            Trans = 112
        };
        inline Order transpose(Order o) {return o == RowMajor ? ColMajor : RowMajor;}
        inline Order transpose(Order o, Transpose t) {return t == NoTrans ? o : transpose(o);}
    }
}


#endif //TENSORSTACK_KERNELS_COMMON_BLAS_H
