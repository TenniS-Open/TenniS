//
// Created by kier on 2018/10/16.
//

#include <module/graph.h>
#include <module/module.h>
#include <global/setup.h>
#include <runtime/workbench.h>
#include <global/operator_factory.h>
#include <utils/ctxmgr.h>
#include <core/tensor_builder.h>
#include <module/menu.h>
#include <utils/box.h>
#include <module/io/fstream.h>

#include <cstring>

class Sum : public ts::Operator {
public:
    using supper = ts::Operator;
    Sum() {
        field("test_field", OPTIONAL);
    }

    virtual void init() {
        supper::init();
    }

    virtual int run(ts::Stack &stack) {
        int input_num = stack.size();
        std::vector<ts::Tensor::Prototype> output;
        this->infer(stack, output);
        TS_AUTO_CHECK(output[0].dtype() == ts::FLOAT32);
        stack.push(output[0]);
        auto &sum = *stack.index(-1);
        std::memset(sum.data<float>(), 0, sum.count() * sizeof(float));
        for (int i = 0; i < input_num; ++i) {
            for (int j = 0; j < sum.count(); ++j) {
                sum.data<float>()[j] += stack.index(i)->data<float>()[j];
            }
        }
        return 1;
    }

    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
        if (stack.size() == 0) throw ts::Exception("Can not sum on empty inputs");
        for (int i = 1; i < stack.size(); ++i) {
            if (stack.index(i)->sizes() != stack.index(0)->sizes()) {
                ts::Exception("Can not sum mismatch size inputs");
            }
        }
        output.resize(1);
        output[0] = stack.index(0)->proto();
        return 1;
    }
};

TS_REGISTER_OPERATOR(Sum, ts::CPU, "sum")

namespace ts {
    Node block(const std::string &prefix, Node in) {
        auto conv = bubble::op(prefix + "/" + "a", "sum", {in});
        auto bn = bubble::op(prefix + "/" + "b", "sum", {conv});
        auto shortcut = bubble::op(prefix + "/" + "out", "sum", {in, bn});
        return shortcut;
    }

    Node block_n_times(const std::string &prefix, Node in, int n) {
        auto blob = in;
        for (int i = 0; i < n; ++i) {
            blob = block(prefix + "/" + "block_" + std::to_string(i), blob);
        }
        return blob;
    }
}

int main()
{
    using namespace ts;
    setup();



    // build graph
    Graph g;
    ctx::bind<Graph> _graph(g);

//    auto a = op::in("input");
//    auto block = block_n_times("block", a, 1000);
//    auto output = op::sum("output", {block});
//    auto a = op::in("a");
//    auto b = op::sum("b", {a});
//    auto c = op::sum("c", {b});
//    auto d = op::sum("d", {a, c});

    auto a = bubble::param("a");
    auto b = bubble::param("b");
    auto data = bubble::data("data", tensor::from<float>(3));

    auto c = bubble::op("c", "sum", {a, b, data});
    
    {
        // test graph
        ts::FileStreamWriter out("test.module.txt");
        serialize_graph(out, g);
        out.close();

        ts::Graph tg;
        ts::FileStreamReader in("test.module.txt");
        externalize_graph(in, tg);
        g = tg;
    }

    // setup module
    std::shared_ptr<Module> m = std::make_shared<Module>();
    m->load(g, {"c"});
    m->sort_inputs({"a", "b"});

    std::cout << "Input nodes:" << std::endl;
    for (auto &node : m->inputs()) {
        std::cout << node.ref<Bubble>().op() << ":" << node.ref<Bubble>().name() << std::endl;
    }

    std::cout << "Output nodes:" << std::endl;
    for (auto &node : m->outputs()) {
        std::cout << node.ref<Bubble>().op() << ":" << node.ref<Bubble>().name() << std::endl;
    }

    // run workbench
    ComputingDevice device(CPU, 0);
    // Workbench bench(device);

    Workbench::shared bench;

    try {
        bench = Workbench::Load(m, device);
    } catch (const Exception &e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    Tensor input_a(FLOAT32, {1});
    Tensor input_b(FLOAT32, {1});

    input_a.data<float>()[0] = 1;
    input_b.data<float>()[0] = 3;

    bench->input("a", input_a);
    bench->input("b", input_b);

    bench->run();

    auto output_c = bench->output("c");

    std::cout << "output: " << output_c.data<float>()[0] << std::endl;
}
