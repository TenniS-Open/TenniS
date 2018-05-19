//
// Created by lby on 2018/2/11.
//

#include <mem/hard_memory.h>
#include <mem/controller.h>
#include <iostream>
#include <global/memory_device.h>

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

    a.data<int>()[0] = 12;

    ts::memcpy(b, a, 123);

    std::cout << b.data<int>()[0] << std::endl;

    std::cout << ts::QueryMemoryDevice(ts::Device(ts::CPU, 0)) << std::endl;

    try {
        ts::QueryMemoryDevice(ts::Device("ARM", 0));
    } catch (const ts::NoMemoryDeviceException &e) {
        std::cout << e.what() << std::endl;
    }


    return 0;
}