//
// Created by lby on 2018/2/11.
//

#include <mem/hard_memory.h>
#include <mem/controller.h>
#include <iostream>

int main()
{
    ts::HardMemory mem({ts::CPU, 0});
    mem.resize(10);
    mem.dispose();

    ts::BaseMemoryController c({ts::CPU, 0});
    ts::Memory a = c.alloc(123);

    ts::Memory b(ts::Device(ts::CPU, 0), 256);

    std::cout << a.size() << std::endl;
    std::cout << b.size() << std::endl;

    return 0;
}