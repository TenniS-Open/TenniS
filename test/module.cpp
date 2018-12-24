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

#include <kernels/cpu/resize2d.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>



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

//    auto a = bubble::param("a");
//    auto b = bubble::param("b");
//    auto b1 = block_n_times("block1", a, 100);
//    auto b2 = block_n_times("block1", b, 100);
//    auto c = bubble::op("c", "sum", {b1, b2});

    /*
    auto a = bubble::param("a");
    auto b = bubble::param("b");
    auto data = bubble::data("data", tensor::from<float>(3));

    auto c = bubble::op("c", "sum", {a, b, data});
    */

    cv::Mat srcimage = cv::imread("/wqy/Downloads/test.png");
    auto a = bubble::param("a");
    auto b = bubble::param("b");
    auto c = bubble::op("c","resize2d",{a,b});

    ts::Shape type_shape = {1};
    Tensor param_type(INT32, type_shape);
    param_type.data<int>()[0] = 0;
    c.ref<Bubble>().set("type",param_type);

    /*
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
    */
    // setup module
    std::shared_ptr<Module> m = std::make_shared<Module>();
    m->load(g, {"c"});
    m->sort_inputs({"a", "b"});

    {
        // test graph
        Module::Save("test.module.txt", m);

        m = Module::Load("test.module.txt");
    }

    /*
    std::cout << "Input nodes:" << std::endl;
    for (auto &node : m->inputs()) {
        std::cout << node.ref<Bubble>().op() << ":" << node.ref<Bubble>().name() << std::endl;
    }

    std::cout << "Output nodes:" << std::endl;
    for (auto &node : m->outputs()) {
        std::cout << node.ref<Bubble>().op() << ":" << node.ref<Bubble>().name() << std::endl;
    }
    */
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

    /*
    Tensor input_a(FLOAT32, {1});
    Tensor input_b(FLOAT32, {1});

    input_a.data<float>()[0] = 1;
    input_b.data<float>()[0] = 3;
    */

    ts::Shape shape = {2,srcimage.rows, srcimage.cols, srcimage.channels()};

    //ts::Shape shape = {1,1,4, 4};
    Tensor input_a(FLOAT32, shape);

    //Tensor input_a(UINT8, shape);
    Tensor input_b(INT32, {4});

    cv::Mat srcimage2 = cv::imread("/wqy/Downloads/test2.png");

    std::cout << "old:" << srcimage.channels() * srcimage.rows * srcimage.cols << std::endl;
    std::cout << "count:" << input_a.count() << std::endl;
    int num = input_a.count();
    float * buffer = new float[num];
    //unsigned char * buffer = new unsigned char[num];

    num = num / 2;
    for(int i=0;i<num; i++) {
         buffer[i ] = srcimage.data[i];
    }

    for(int i=0;i<num; i++) {
         buffer[i + num ] = srcimage2.data[i];
    }

    memcpy(input_a.data<float>(), buffer, num * sizeof(float) * 2);
    delete [] buffer;
 
    input_b.data<int>()[0] = -1;
    input_b.data<int>()[1] = 400;
    input_b.data<int>()[2] = 400;
    input_b.data<int>()[3] = -1;


    bench->input("a", input_a);
    bench->input("b", input_b);

    bench->run();

    auto output_c = bench->output("c");
    cv::Mat dstimage(400,400,CV_32FC3,output_c.data<float>());
    cv::Mat dstimage2(400,400,CV_32FC3,output_c.data<float>() + 400 * 400 * 3);

    //cv::Mat dstimage(400,400,CV_8UC3,output_c.data<unsigned char>());
    //cv::Mat dstimage2(400,400,CV_8UC3,output_c.data<unsigned char>() + 400 * 400 * 3);
    cv::imwrite("/tmp/mm3.png", dstimage);
    cv::imwrite("/tmp/mm4.png", dstimage2);

    //std::cout << "output: " << output_c.data<float>()[0] << std::endl;
}
