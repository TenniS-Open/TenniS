//
// Created by kier on 2018/10/30.
//

#include <iostream>
#include <core/tensor_converter.h>
#include <global/setup.h>

int main() {
    using namespace ts;
    setup();

    std::string str = "abcd";

    auto a = tensor::from(str);

    std::cout << str << " vs. " << tensor::to_string(a) << std::endl;

}
