//
// Created by kier on 2018/10/16.
//

#include <module/graph.h>
#include <module/module.h>
#include <global/setup.h>
#include <runtime/workbench.h>
#include <global/operator_factory.h>

class Add : public ts::Operator {
public:
    virtual int run(ts::Stack &stack) {
        return 0;
    }
    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
        return 0;
    }
};

TS_REGISTER_OPERATOR(Add, ts::CPU, "add")

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
    std::shared_ptr<Module> m = std::make_shared<Module>();
    m->load(g);

    std::cout << "Input nodes:" << std::endl;
    for (auto &node : m->inputs()) {
        std::cout << node.ref<OP>().op << ":" << node.ref<OP>().name << std::endl;
    }

    std::cout << "Output nodes:" << std::endl;
    for (auto &node : m->outputs()) {
        std::cout << node.ref<OP>().op << ":" << node.ref<OP>().name << std::endl;
    }

    // run workbench
    ComputingDevice device(CPU, 0);
    // Workbench bench(device);

    auto bench = Workbench::Load(m, device);
}
