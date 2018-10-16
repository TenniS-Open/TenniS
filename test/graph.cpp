//
// Created by kier on 2018/10/16.
//

#include <module/graph.h>
#include <iostream>

template <typename T>
class Add {
public:
    T forward(const std::vector<ts::Node> &inputs) {
        T sum = 0;
        for (auto &input : inputs) {
            sum += input.ref<int>();
        }
        return sum;
    }
};

int main()
{
    ts::Graph g;
    auto a = g.make<int>(10);
    auto b = g.make<int>(20);
    auto c = g.make<Add<int>>();
    ts::Node::Link(c, {a, b});
    auto sum = c.ptr<Add<int>>()->forward(c.inputs());
    std::cout << "sum: " << sum << std::endl;
}