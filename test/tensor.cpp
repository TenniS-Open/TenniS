//
// Created by seeta on 2018/5/25.
//

#include <core/tensor.h>
#include <core/type.h>
#include <iostream>
#include <core/scan.h>

int main() {
    using namespace ts;
    Tensor a(FLOAT32, {2, 3, 4});

    float *data = a.data<float>();
    for (int i = 0; i < a.count(); ++i) {
        data[i] = i;
    }

    BaseItrator it(&a);

//    while (true)
//    {
//        if (it.data() == it.end()) break;
//
//        std::cout << *it.data<float>() << std::endl;
//
//        it.next();
//    }

    std::cout << *reinterpret_cast<float*>(it.at({1,1,1})) << std::endl;
    std::cout << *reinterpret_cast<float*>(it.at(1, 1, 0)) << std::endl;

    auto b = Scan(3, 1);
    auto c = Scan(3, -3);
    auto d = Scan(3, -3);
    auto e = Scan(2, c);
    Scan::Group g(Scan(0, 12), Scan(0, 1));
    auto gg = Scan(3, std::move(g));
    auto f = Scan(1, {Scan(1, 0), std::move(b), Scan(3, 2), std::move(gg), std::move(e)});
    std::cout << std::endl;

    std::cout << f.update() << std::endl;

    auto t = f.clone();
    std::cout << std::endl;

    std::cout << std::endl;

    std::cout << std::endl;

    Loop loop;
    loop.bind(t);

    while (true) {
        auto step = loop.next();
        if (step == Loop::finish) break;
        std::cout << step << " ";
    }

    std::cout << std::endl;

    return 0;
}

