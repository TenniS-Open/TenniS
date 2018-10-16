//
// Created by lby on 2018/3/12.
//

#include "global/setup.h"
#include "kernels/cpu/memory_cpu.h"

namespace ts {
    void setup() {
        // may do some setup
        cpu_allocator(0, 0, nullptr);
    }
}

