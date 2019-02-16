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
#include <core/tensor_builder.h>

#include <cstring>
#include <map>

#include <kernels/cpu/add.h>
#include <kernels/cpu/add_bias.h>
#include <kernels/cpu/batch_norm.h>
#include <kernels/cpu/batch_scale.h>
#include <kernels/cpu/concat.h>
#include <kernels/cpu/conv2d.h>
//#include <kernels/cpu/conv2d_v2.h>
#include <kernels/cpu/depthwise_conv2d.h>
//#include <kernels/cpu/depthwise_conv2d_v2.h>
#include <kernels/cpu/dimshuffle.h>
#include <kernels/cpu/div.h>
#include <kernels/cpu/flatten.h>
#include <kernels/cpu/fused_batch_norm.h>
#include <kernels/cpu/inner_prod.h>
#include <kernels/cpu/mul.h>
#include <kernels/cpu/pad.h>
#include <kernels/cpu/pooling2d.h>
#include <kernels/cpu/pooling2d_v2.h>
#include <kernels/cpu/prelu.h>
#include <kernels/cpu/prewhiten.h>
#include <kernels/cpu/relu.h>
#include <kernels/cpu/relu_max.h>
#include <kernels/cpu/reshape.h>
#include <kernels/cpu/resize2d.h>
#include <kernels/cpu/sigmoid.h>
#include <kernels/cpu/softmax.h>
#include <kernels/cpu/sub.h>
#include <kernels/cpu/to_float.h>
#include <kernels/cpu/transpose.h>
#include "backend/mxnet/pooling2d_padding.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "backend/name.h"

using namespace ts;
using namespace ts::mxnet;
using namespace cpu;
TS_REGISTER_OPERATOR(Add, CPU, name::layer::add())
TS_REGISTER_OPERATOR(AddBias, ts::CPU, ts::name::layer::add_bias())
TS_REGISTER_OPERATOR(BatchNorm, ts::CPU, ts::name::layer::batch_norm())
TS_REGISTER_OPERATOR(BatchScale, ts::CPU, ts::name::layer::batch_scale())
TS_REGISTER_OPERATOR(Conv2d, ts::CPU, ts::name::layer::conv2d())
//TS_REGISTER_OPERATOR(Conv2d_V2, ts::CPU, ts::name::layer::conv2d_v2())
//TS_REGISTER_OPERATOR(Depthwise_Conv2d_V2, ts::CPU, ts::name::layer::depthwise_conv2d_v2())
TS_REGISTER_OPERATOR(Dimshuffle, ts::CPU, ts::name::layer::dimshuffle())
TS_REGISTER_OPERATOR(Div, ts::CPU, ts::name::layer::div())
TS_REGISTER_OPERATOR(Flatten, ts::CPU, ts::name::layer::flatten())
TS_REGISTER_OPERATOR(Fused_Batch_Norm, ts::CPU, ts::name::layer::fused_batch_norm())
TS_REGISTER_OPERATOR(Inner_Prod, ts::CPU, ts::name::layer::inner_prod())
TS_REGISTER_OPERATOR(Mul, ts::CPU, ts::name::layer::mul())
TS_REGISTER_OPERATOR(Pad, ts::CPU, ts::name::layer::pad())
TS_REGISTER_OPERATOR(Reshape, ts::CPU, ts::name::layer::reshape())
TS_REGISTER_OPERATOR(Resize2d, ts::CPU, ts::name::layer::resize2d())
TS_REGISTER_OPERATOR(To_Float, ts::CPU, ts::name::layer::to_float())
TS_REGISTER_OPERATOR(Transpose, ts::CPU, ts::name::layer::transpose())
TS_REGISTER_OPERATOR(Pooling2dV2, ts::CPU, name::layer::pooling2d_v2())
TS_REGISTER_OPERATOR(Pooling2d, ts::CPU, name::layer::pooling2d())
TS_REGISTER_OPERATOR(Prelu, ts::CPU, name::layer::prelu())
TS_REGISTER_OPERATOR(Relu, ts::CPU, name::layer::relu())
TS_REGISTER_OPERATOR(ReluMax, ts::CPU, name::layer::relu_max())
TS_REGISTER_OPERATOR(Sigmoid, ts::CPU, name::layer::sigmoid())
TS_REGISTER_OPERATOR(Concat, ts::CPU, name::layer::concat())
TS_REGISTER_OPERATOR(Softmax, ts::CPU, name::layer::softmax())
TS_REGISTER_OPERATOR(Depthwise_Conv2d, ts::CPU, ts::name::layer::depthwise_conv2d())
TS_REGISTER_OPERATOR(PreWhiten, ts::CPU, "prewhiten")
TS_REGISTER_OPERATOR(Pooling2dPadding, CPU, name::layer::mx_pooling2d_padding())

namespace ts {
    Node block(const std::string &prefix, Node in) {
        auto conv = bubble::op(prefix + "/" + "a", "sum", { in });
        auto bn = bubble::op(prefix + "/" + "b", "sum", { conv });
        auto shortcut = bubble::op(prefix + "/" + "out", "sum", { in, bn });
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
        m_duration = std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_start);

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

inline int offset(const std::vector<int>& indices, const std::vector<int>& shape_)
{
    int offset = 0;
    for (int i = 0; i < shape_.size(); ++i)
    {
        offset *= shape_[i];
        if (indices.size() > i)
        {
            offset += indices[i];
        }
    }
    return offset;
}

class GrandTruth {
public:
    std::map<std::string, std::vector<float>> values;

    void read(const std::string &filename) {
        std::ifstream gt(filename);
        std::string name;
        std::string blob;
        while (true) {
            std::getline(gt, name);
            std::getline(gt, blob);

            if (name.empty() || blob.empty()) {
                break;
            }

            std::vector<float> vec;
            std::istringstream iss(blob);
            float f;
            while (iss >> f) {
                vec.push_back(f);
            }
            values.insert(std::make_pair(name, std::move(vec)));
        }
    }
};

int main()
{
    using namespace ts;
    setup();

    std::shared_ptr<Module> m = std::make_shared<Module>();
    //m = Module::Load("../model/test.tsm");
    m = Module::Load("/Users/seetadev/Documents/SDK/CLion/TensorStack/python/test/SeetaFaceRecognizer5.0.RN30.light.1out.tsm");
    std::cout << "Input nodes:" << std::endl;
    for (auto &node : m->inputs()) {
        std::cout << node.bubble().op() << ":" << node.bubble().name() << std::endl;
    }

    std::cout << "Output nodes:" << std::endl;
    for (auto &node : m->outputs()) {
        std::cout << node.bubble().op() << ":" << node.bubble().name() << std::endl;
    }

    // run workbench
    ComputingDevice device(CPU, 0);
    // Workbench bench(device);

    Workbench::shared bench;

    try {
        bench = Workbench::Load(m, device);
        bench = bench->clone();
    }
    catch (const Exception &e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    cv::Mat input_img = cv::imread("input_test.png");
    ts::Shape shape = { 1,3, input_img.rows,input_img.cols };
    Tensor input_param(UINT8, shape);
    int num = input_param.count();
    uchar * buffer = new uchar[num];
    uchar* input_data = input_img.data;
    std::vector<int> index_vector(4, 0);
    int trans_index = 0;
    for (int n = 0; n < shape[0]; n++)
    {
        index_vector[0] = n;
        for (int h = 0; h < shape[2]; h++)
        {
            index_vector[2] = h;
            for (int w = 0; w < shape[3]; w++)
            {
                index_vector[3] = w;
                for (int c = 0; c < shape[1]; c++)
                {
                    index_vector[1] = c;
                    uchar val = input_data[trans_index++];
                    int buffer_index = offset(index_vector, shape);
                    buffer[buffer_index] = val;
                }
            }
        }
    }
    std::memcpy(input_param.data<uchar>(), buffer, num * sizeof(uchar));
    delete[] buffer;

//	ts::Shape shape = { 1,3, 248,248 };
//	Tensor input_param(FLOAT32, shape);

//	for (int i = 0; i < input_param.count(); i++)
//	{
//		input_param.data<float>()[i] = 100;
//	}

    float count_time = 0;
    int num_count = 100;
    for (size_t i = 0; i < num_count; i++)
    {
        bench->input("_input", input_param);
        {
            //time_log _log(ts::LOG_INFO, "Spent ");
            time_point start;
            time_point end;
            start = std::chrono::system_clock::now();

            bench->run();

            end = std::chrono::system_clock::now();
            std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            count_time += duration.count() / 1000.0;
        }
    }
    std::cout << "spent time: " << count_time / num_count << "ms" << std::endl;

    GrandTruth gt;
    gt.read("gt.txt");

    std::cout << "++++++++++++++++++++++++++++++" << std::endl;

    for (auto &pair : gt.values) {
        std::cout << "GT got: " << pair.first << ", size = " << pair.second.size() << std::endl;
    }

    // std::ofstream f("test.txt");
    for (auto &node : m->outputs()) {
        std::cout << "==============================" << std::endl;
        std::cout << node.bubble().op() << ":" << node.bubble().name() << std::endl;
        auto data = bench->output(node.bubble().name());
        std::vector<int> vec = data.sizes();

        std::cout << vec.size() << ",count:" << data.count() << std::endl;
        for (int i = 0; i<vec.size(); i++)
            std::cout << vec[i] << ",";
        std::cout << std::endl;

        auto name_gt = gt.values.find(node->name());
        if (name_gt == gt.values.end()) continue;



        float *lhs = data.data<float>();
        float *rhs = name_gt->second.data();
        auto count = data.count();

        float max = 0;
        float sum = 0;

        for (int i = 0; i < count; ++i) {
            auto diff = std::fabs(lhs[i] - rhs[i]);
            sum  += diff;
            if (diff > max) max = diff;
        }

        auto avg = sum / count;

        std::cout << "max = " << max << ", avg = " << avg << "." << std::endl;
    }
    return 0;
}
