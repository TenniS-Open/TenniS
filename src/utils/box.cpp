//
// Created by kier on 2018/10/31.
//

#include "utils/box.h"
#include <algorithm>
#include <memory>

namespace ts {
    template<typename T>
    static inline T min(T a, T b, T c) {
        return std::min<T>(std::min<T>(a, b), c);
    }

    int edit_distance(const std::string &lhs, const std::string &rhs) {
        const size_t M = lhs.length();  // rows
        const size_t N = rhs.length();  // cols

        if (M == 0) return int(N);
        if (N == 0) return int(M);

        std::unique_ptr<int[]> dist(new int[M * N]);
#define __EDIT_DIST(m, n) (dist[(m) * N + (n)])
        __EDIT_DIST(0, 0) = lhs[0] == rhs[0] ? 0 : 2;
        for (size_t n = 1; n < N; ++n) {
            __EDIT_DIST(0, n) = __EDIT_DIST(0, n - 1) + 1;
        }
        for (size_t m = 1; m < M; ++m) {
            __EDIT_DIST(m, 0) = __EDIT_DIST(m - 1, 0) + 1;
        }
        for (size_t m = 1; m < M; ++m) {
            for (size_t n = 1; n < N; ++n) {
                if (lhs[m] == rhs[n]) {
                    __EDIT_DIST(m, n) = min(
                            __EDIT_DIST(m - 1, n),
                            __EDIT_DIST(m, n - 1),
                            __EDIT_DIST(m - 1, n - 1));
                } else {
                    __EDIT_DIST(m, n) = min(
                            __EDIT_DIST(m - 1, n) + 1,
                            __EDIT_DIST(m, n - 1) + 1,
                            __EDIT_DIST(m - 1, n - 1) + 2);
                }
            }
        }
        return dist[M * N - 1];
#undef __EDIT_DIST
    }
}
