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
        stack.push(output[0], memory_device()); // Notice push tensor device, default is DeviceContext.memory_device
        auto sum = stack.index(-1);
        auto sum_ptr = sum->view(ts::MemoryDevice(ts::CPU)).data<float>();    // Get CPU data ptr
        std::memset(sum_ptr, 0, sum->count() * sizeof(float));
        for (int i = 0; i < input_num; ++i) {
            auto input_ptr = stack.index(i)->view(memory_device()).data<float>();   // Get CPU data ptr
            for (int j = 0; j < sum->count(); ++j) {
                sum_ptr[j] += input_ptr[j];
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

class time_log {
public:
    using self = time_log;

    using microseconds = std::chrono::microseconds;
    using system_clock = std::chrono::system_clock;
    using time_point = decltype(system_clock::now());

    explicit time_log(ts::LogLevel level, const std::string &header = "") :
            m_duration(0) {
        m_level = level;
        m_header = header;
        m_start = system_clock::now();
    }

    ~time_log() {
        m_end = system_clock::now();
        m_duration = std::chrono::duration_cast<microseconds>(m_end - m_start);

        std::ostringstream oss;
        ts::LogStream(m_level) << m_header << m_duration.count() / 1000.0 << "ms";
    }

    time_log(const self &) = delete;
    self &operator=(const self &) = delete;

private:
    ts::LogLevel m_level;
    std::string m_header;
    microseconds m_duration;
    time_point m_start;
    time_point m_end;
};

int main()
{
    using namespace ts;
    setup();

    GlobalLogLevel(LOG_DEBUG);



    // build graph
    Graph g;
    ctx::bind<Graph> _graph(g);

//    auto a = bubble::param("a");
//    auto b = bubble::param("b");
//    auto b1 = block_n_times("block1", a, 100);
//    auto b2 = block_n_times("block1", b, 100);
//    auto c = bubble::op("c", "sum", {b1, b2});

    auto a = bubble::param("a");
    auto b = bubble::param("b");
    auto data = bubble::data("data", tensor::from<float>(3), CPU);
    auto c = bubble::op("c", "sum", {a, b, data});

    {
        // test graph
        ts::FileStreamWriter out("test.graph.txt");
        serialize_graph(out, g);
        out.close();

        ts::Graph tg;
        ts::FileStreamReader in("test.graph.txt");
        externalize_graph(in, tg);
        g = tg;
    }

    // setup module
    std::shared_ptr<Module> m = std::make_shared<Module>();
    m->load(g, {"c"});
    m->sort_inputs({"a", "b"});

    {
        // test graph
        // Module::Save("test.module.txt", m);

        m = Module::Load("/Users/seetadev/Documents/SDK/CLion/TensorStack/python/test/test.module.txt");
    }

    std::cout << "Input nodes:" << std::endl;
    for (auto &node : m->inputs()) {
        std::cout << node->op() << ":" << node->name() << std::endl;
    }

    std::cout << "Output nodes:" << std::endl;
    for (auto &node : m->outputs()) {
        std::cout << node->op() << ":" << node->name() << std::endl;
    }

    // run workbench
    ComputingDevice device(CPU, 0);
    // Workbench bench(device);

    Workbench::shared bench;

    try {
        bench = Workbench::Load(m, device);
        bench = bench->clone();
    } catch (const Exception &e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    Tensor input_a(FLOAT32, {});
    Tensor input_b(FLOAT32, {});

    input_a.data<float>()[0] = 1;
    input_b.data<float>()[0] = 3;

    bench->do_profile(true);

    bench->input("a", input_a);
    bench->input("b", input_b);
    bench->run();

    {
        time_log _log(ts::LOG_INFO, "Spent ");

        for (int i = 0; i < 100; ++i) {
            bench->run();
        }

    }

    bench->profiler().log(std::cout);

    auto output_c = bench->output("c");

    std::cout << "output shape: " << to_string(output_c.sizes()) << std::endl;

    std::cout << "output: " << output_c.data<float>()[0] << std::endl;
}
