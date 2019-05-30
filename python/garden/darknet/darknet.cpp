//
// Created by kier on 2019-05-28.
//

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "api/cpp/tensorstack.h"

int main() {
    using namespace ts::api;
    Device device("cpu", 0);

//    cv::Mat cvimage = cv::imread("/Users/seetadev/Documents/SDK/CLion/darknet/bin/data/dog.jpg");
    cv::Mat cvimage = cv::imread("/Users/seetadev/Documents/SDK/CLion/darknet/bin/data/000195.jpg");
    std::string model = "/Users/seetadev/Documents/SDK/CLion/TensorStack/python/test/yolov3.tsm";
    std::vector<std::string> label = {
            "bai_sui_shan",
            "cestbon",
            "cocacola",
            "jing_tian",
            "pepsi_cola",
            "sprite",
            "starbucks_black_tea",
            "starbucks_matcha",
            "starbucks_mocha",
            "vita_lemon_tea",
            "vita_soymilk_blue",
            "wanglaoji_green",
    };

    std::string gt_root = "/Users/seetadev/Documents/SDK/CLion/darknet/bin/layer_blobs";

    Workbench bench(device);
    bench.setup_context();
    bench.setup_device();
    bench.set_computing_thread_number(4);

    ImageFilter filter(device);
    filter.channel_swap({2, 1, 0});
    filter.to_float();
    filter.scale(1 / 255.0);
    filter.letterbox(416, 416, 0.5);    // darknet pre-processor
    filter.to_chw();

    bench.setup(bench.compile(Module::Load(model)));

    Tensor tensor = tensor::build(UINT8, {1, cvimage.rows, cvimage.cols, cvimage.channels()}, cvimage.data);
//    tensor = filter.run(tensor);

//    tensor = tensor::load(gt_root + "/input.t");

    bench.bind_filter(0, filter);
    bench.input(0, tensor);

    bench.run();


    struct BBox {
        float x;
        float y;
        float w;
        float h;
        float scale;
        float label;
    };
    auto output_count = bench.output_count();

    std::stringstream oss;
    for (int i = 0; i < output_count; ++i) {
        auto output = bench.output(i);
        output = tensor::cast(FLOAT32, output);

        int N = output.size(0);

        for (int n = 0; n < N; ++n) {
            auto box = output.data<BBox>(n);
            box.x *= float(cvimage.cols);
            box.y *= float(cvimage.rows);
            box.w *= float(cvimage.cols);
            box.h *= float(cvimage.rows);

            srand(int(box.label));
            auto r = rand() % 192 + 64;
            auto g = rand() % 192 + 64;
            auto b = rand() % 192 + 64;

            cv::rectangle(cvimage, cv::Rect(box.x, box.y, box.w, box.h), CV_RGB(r, g, b), 2);

            oss.str("");
            if (label.empty()) {
                oss << int(box.label) << ": " << int(std::round(box.scale) * 100) << "%";
            } else {
                oss << label[int(box.label)] << ": " << int(std::round(box.scale * 100)) << "%";
            }

            std::cout << oss.str() << std::endl;
            cv::putText(cvimage, oss.str(), cv::Point(box.x, box.y), CV_FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(r - 32, g - 32, b - 32));
        }
    }

    cv::imshow("Test", cvimage);
    cv::waitKey();

    return 0;
}