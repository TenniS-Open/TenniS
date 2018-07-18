//
// Created by lby on 2018/2/11.
//

#include <core/hard_memory.h>
#include <core/controller.h>
#include <iostream>
#include <global/memory_device.h>

#include <map>
#include <unordered_map>

int main()
{
    ts::HardMemory mem({ts::CPU, 0});
    try {
        mem.resize(10000000000000000000UL);
    } catch (const ts::Exception &e) {
        std::cout << e.what() << std::endl;
    }
    mem.resize(10);
    mem.data<int>()[0] = 10;
    std::cout << mem.data<int>()[0] << std::endl;
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

    std::map<ts::Device, int> aa;
    std::unordered_map<ts::Device, int> bb;

    return 0;
}