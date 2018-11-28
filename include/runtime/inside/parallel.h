//
// Created by kier on 2018/11/28.
//

#ifndef TENSORSTACK_RUNTIME_INSIDE_PARALLEL_H
#define TENSORSTACK_RUNTIME_INSIDE_PARALLEL_H


#include "thread_pool.h"
#include "utils/ctxmgr.h"
#include "utils/box.h"
#include "utils/log.h"

namespace ts {
    inline ThreadPool *try_parallel(int task_number) {
        if (task_number <= 1) return nullptr;
        auto gun = ctx::ptr<ThreadPool>();
        if (gun != nullptr && gun->size() > 1) return gun;
        return nullptr;
    }

    inline void parallel_run(const std::function<void(int, int)> &range_solver, int begin, int end, bool joinable = true) {
        auto parallel_gun = ts::try_parallel(end - begin);
        if (parallel_gun) {
            auto parallel_ranges = ts::split_bins(begin, end, int(parallel_gun->size()));
            for (auto &parallel_range : parallel_ranges) {
                parallel_gun->run([range_solver, parallel_range](int){
                    range_solver(parallel_range.first, parallel_range.second);
                });
            }
            if (joinable) {
                parallel_gun->join();
            }
        } else {
            range_solver(begin, end);
        }
    }

    inline void parallel_sync() {
        auto gun = ctx::ptr<ThreadPool>();
        if (gun) gun->join();
    }
}

/**
 * Parallel for loop
 * @param var_loop_value loop value name
 * @param var_loop_begin loop begin value
 * @param var_loop_end loop end value
 * @note the TS_PARALLEL_XXX block do NOT support nest
 * @note The input parameters over 3 are the closure value in parallel run
 * Usage:
 * ```
 * TS_PARALLEL_FOR_BEGIN(i, 0, 10)
 *     std::cout << i << std::endl;
 * TS_PARALLEL_FOR_END()
 * TS_PARALLEL_SYNC
 * ```
 * equal to codes:
 * ```
 * for (int i = 0; i < 10; ++i) {
 *     std::cout << i << std::endl;
 * }
 * ```
 * , but in parallel
 * @note remeber use TS_PARALLEL_SYNC after every parallel task should sync of finish
 */
#define TS_PARALLEL_FOR_BEGIN(var_loop_value, var_loop_begin, var_loop_end, ...) \
{ \
    int __ts_parralel_begin = int(var_loop_begin); \
    int __ts_parralel_end = int(var_loop_end); \
    auto __ts_parralel_solver = [&, ## __VA_ARGS__](int begin, int end) -> void { \
        int var_loop_value = begin; \
        for (; var_loop_value < end; ++var_loop_value) { \


/**
 * @note TS_PARALLEL_FOR_END can parse an bool value, mean the parallel tasks if is joinable
 * @see TS_PARALLEL_FOR_BEGIN
 */
#define TS_PARALLEL_FOR_END(...) \
        } \
    }; \
    ts::parallel_run(__ts_parralel_solver, __ts_parralel_begin, __ts_parralel_end, ## __VA_ARGS__); \
}

/**
 * @see TS_PARALLEL_FOR_BEGIN
 */
#define TS_PARALLEL_SYNC \
ts::parallel_sync();


#endif //TENSORSTACK_RUNTIME_INSIDE_PARALLEL_H
