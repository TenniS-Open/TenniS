//
// Created by seeta on 2018/6/28.
//

#include <stack/stack.h>
#include <iostream>
#include <stack/workbench.h>

int main() {

    ts::Stack ss(ts::Device(ts::CPU));
    ss.push(ts::TYPE::INT8, {1});
    std::cout << ss.size() << std::endl;
    ss.push(-1);
    ss.push(-1);
    ss.push(-1);
    ss.push(1);
    ss.pop(2);
    std::cout << ss.size() << std::endl;
    // ss.rebase(2);
    ss.push_base(0);
    std::cout << ss.size() << std::endl;
    ss.pop_base();
    std::cout << ss.size() << std::endl;


}