//
// Created by kier on 2018/12/21.
//

#ifndef TENSORSTACK_KERNELS_COMMON_OPENMP_H
#define TENSORSTACK_KERNELS_COMMON_OPENMP_H

#ifdef TS_USE_OPENMP
#include <omp.h>
#include "runtime/runtime.h"
#include <algorithm>

#define TS_OPENMP_BLOCK_SIZE 10240

#endif

namespace ts {
    inline int openmp_threads(const int task_number) {
#ifdef TS_USE_OPENMP
        auto threads = std::min<int>(std::max<int>(task_number / TS_OPENMP_BLOCK_SIZE, 1), omp_get_num_procs());
        auto runtime = ctx::ptr<RuntimeContext>();
        if (runtime == nullptr) return threads;
        return std::min(runtime->get_computing_thread_number(), threads);
#else
        return 1;
#endif
    }

    inline int openmp_thread_id() {
#ifdef TS_USE_OPENMP
        return omp_get_thread_num();
#else
        return 0;
#endif
    }
}





#endif //TENSORSTACK_KERNELS_COMMON_OPENMP_H
