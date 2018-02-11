//
// Created by lby on 2018/2/11.
//

#include <mem/hard_memory.h>

int main()
{
    ts::HardMemory mem({ts::CPU, 0});
    mem.resize(10);
    mem.dispose();

    return 0;
}