//
// Created by kier on 2018/11/28.
//

#include "runtime/inside/parallel.h"

#include "utils/log.h"

void test_parallel() {
    for (int i = 0; i < 10; ++i) {
        TS_PARALLEL_FOR_BEGIN(j, 0, 10, i)
                    TS_LOG_INFO << i * 10 + j;
        TS_PARALLEL_FOR_END()
    }
}

int main() {

    ts::ThreadPool pool(16);

    ts::ctx::bind<ts::ThreadPool> _bind(pool);

    test_parallel();

    return 0;
}

