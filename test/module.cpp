//
// Created by kier on 2018/10/16.
//

#include <module/graph.h>
#include <module/module.h>
#include <global/setup.h>

int main()
{
    using namespace ts;
    setup();

    // build graph
    Graph g;
    auto a = g.make<OP>(OP::IN, "a");
    auto b = g.make<OP>(OP::IN, "b");
    auto c = g.make<OP>("add", "c");
    Node::Link(c, {a, b});

    // setup module
    Module m;
    m.load(g);

    std::cout << "Input nodes:" << std::endl;
    for (auto &node : m.inputs()) {
        std::cout << node.ref<OP>().op << ":" << node.ref<OP>().name << std::endl;
    }

    std::cout << "Output nodes:" << std::endl;
    for (auto &node : m.outputs()) {
        std::cout << node.ref<OP>().op << ":" << node.ref<OP>().name << std::endl;
    }

    // run workbench
}
