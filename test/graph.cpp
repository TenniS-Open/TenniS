//
// Created by kier on 2018/10/16.
//

#include <module/graph.h>
#include <module/module.h>
#include <module/menu.h>
#include <runtime/workbench.h>
#include <iostream>

template <typename T>
class Add {
public:
    T forward(const std::vector<ts::Node> &inputs) {
        T sum = 0;
        for (auto &input : inputs) {
            // sum += input.ref<int>();
        }
        return sum;
    }
};

template <typename T>
inline std::ostream &operator<<(std::ostream &out, const Add<T> &node) {
    return out << "Add";
}

int main()
{
    using namespace ts;
    ComputingDevice device;


    Graph g;
    g.setup_context();

    auto shape = bubble::data("shape", tensor::build(INT64, {1, 0, 9}));
    auto data = bubble::param("data");
    auto reshape = bubble::op("reshape", "_reshape_v2", {data, shape});

    auto module = std::make_shared<Module>();
    module->load(g, {reshape});

    auto bench = Workbench::Load(module, device);

    auto input = Tensor(INT32, {1, 2, 3, 3});
    bench->input(0, input);
    bench->run();
    auto output = bench->output(0);

    TS_LOG_INFO << output.proto();

    return 0;
}
