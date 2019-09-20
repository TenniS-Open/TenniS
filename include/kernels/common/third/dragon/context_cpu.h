//
// Created by kier on 2019/9/6.
//

#ifndef TENSORSTACK_THIRD_DRAGON_CONTEXT_CPU_H
#define TENSORSTACK_THIRD_DRAGON_CONTEXT_CPU_H

#include <cstdint>
#include <cstdlib>

#include "context.h"

namespace ts {
    namespace dragon {
        class Workspace;

        class CPUContext : public BaseContext {
        public:
            using self = CPUContext;
            using supper = BaseContext;

            CPUContext(Workspace *ws) : supper(ws) {}

            template<class DstContext, class SrcContext>
            static void Memcpy(
                    size_t nbytes,
                    void *dst,
                    const void *src) {
                std::memcpy(dst, src, nbytes);
            }

            template<typename T, typename DstContext, typename SrcContext>
            void Copy(int n, T *dst, const T *src) {
                if (dst == src) return;
                if (std::is_fundamental<T>::value) {
                    Memcpy<DstContext, SrcContext>(n * sizeof(T), dst, src);
                } else {
                    for (int i = 0; i < n; ++i) dst[i] = src[i];
                }
            }
        };
    }
}

#endif //TENSORSTACK_THIRD_DRAGON_CONTEXT_CPU_H
