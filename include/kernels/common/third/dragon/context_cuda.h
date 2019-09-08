//
// Created by kier on 2019/9/6.
//

#ifndef TENSORSTACK_THIRD_DRAGON_CONTEXT_CUDA_H
#define TENSORSTACK_THIRD_DRAGON_CONTEXT_CUDA_H

#include <cstdint>
#include <cstdlib>

#include "context.h"

namespace ts {
    namespace dragon {
        class Workspace;

#ifdef TS_USE_CUDA

        class CUDAContext : public BaseContext {
        public:
            using self = CUDAContext;
            using supper = BaseContext;

            CUDAContext(Workspace *ws) : supper(ws) {}

            template<typename T, typename TD, typename FD>
            void Copy(size_t count, T *dst, const T *src);
        };

#else
        class CUDAContext : public BaseContext {};
#endif
    }
}

#endif //TENSORSTACK_THIRD_DRAGON_CONTEXT_CUDA_H
