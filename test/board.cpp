//
// Created by kier on 2019/1/29.
//

#include <board/board.h>
#include <board/profiler.h>
#include <utils/ctxmgr_lite.h>

#include <iostream>

uint64_t test_time(uint64_t n) {
    auto _timer = ts::profiler_serial_timer("test(%04d):in");
    uint64_t sum = 0;
    for (uint64_t i = 0; i < n; ++i) {
        sum += i * i;
    }
    return sum;
}

int main() {
    using namespace ts;
    Profiler profiler;

    Board<float> &board = profiler.board();


    ctx::bind<Profiler> _bind(profiler);
    {
        auto _timer = ts::profiler_serial_timer("test(%04d):out");
        test_time(1000000);
    }

    for (auto &name_value : board) {
        std::cout << name_value.first << ": ";
        for (auto &datum : name_value.second) {
            std::cout << datum << ", ";
        }
        std::cout << "avg = " << name_value.second.avg() << "ms.";
        std::cout << std::endl;
    }

    return 0;
}

