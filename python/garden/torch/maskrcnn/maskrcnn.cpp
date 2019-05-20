//
// Created by kier on 19-5-20.
//

#include "plugin.hpp"


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

struct RegOP {
    RegOP() {
        api::RegisterOperator<AnchorGenerator>("cpu", "maskrcnn:anchor_generator");
        api::RegisterOperator<BoxSelector>("cpu", "maskrcnn:box_selector");
        api::RegisterOperator<PostProcessor>("cpu", "maskrcnn:post_processor");
        api::RegisterOperator<ROIPooler>("cpu", "maskrcnn:pooler");

    }
} _reg_op;

using namespace ts::api;

class MaskRCNN {
public:
    class BindingBox {
    public:
        BindingBox() = default;
        BindingBox(float x1, float y1, float x2, float y2, float score, int32_t label)
            : x1(x1), y1(y1), x2(x2), y2(y2), score(score), label(label) {}

        float x1 = 0;
        float y1 = 0;
        float x2 = 0;
        float y2 = 0;
        float score = 0;
        int32_t label = 0;
    };
public:
    const int SIZE_DIVISIBILITY = 32;
    float threshold = 0.7f;

    std::vector<std::string> categories;
public:
    MaskRCNN(const std::string &path, const std::string &device, int id = 0) {
        Device computing_device(device, 0);
        auto module = Module::Load(path);
        bench = Workbench::Load(module, computing_device);
        bench.setup_context();
        bench.setup_device();
        bench.set_computing_thread_number(8);

        filter = ImageFilter(computing_device);
        filter.to_float();
        filter.resize(600);
        filter.sub_mean({102.9801, 115.9465, 122.7717});
        filter.to_chw();
    }

    struct XYXY {
    public:
        float x1;
        float y1;
        float x2;
        float y2;
    };

    /**
     *
     * @param cvimage ready to detect image
     * @return
     * Net output is:
     * list of [Int32[2], Float[N, 4], Float[N], Int32[N]], N means the number of result for
     * [Image size(wh), box result in 'xyxy' mode, socres, labels]
     */
    std::vector<BindingBox> run(const cv::Mat &cvimage) {
        bench.setup_context();
        bench.setup_device();

        auto orignal_width = cvimage.cols;
        auto orignal_height = cvimage.rows;

        auto input_tensor = tensor::build(UINT8, {cvimage.rows, cvimage.cols, cvimage.channels()}, cvimage.data);

        auto transed_image = filter.run(input_tensor);

        auto stride = SIZE_DIVISIBILITY;
        auto transed_height = transed_image.size(2);
        auto transed_width = transed_image.size(3);
        auto fixed_height = int(std::ceil(float(transed_height) / stride) * stride);
        auto fixed_width = int(std::ceil(float(transed_width) / stride) * stride);

        auto padding = tensor::build(INT32, {4, 2}, {0, 0, 0, 0, 0, fixed_height - transed_height, 0, fixed_width - transed_width});
        auto fixed_image = intime::pad(transed_image, padding);

        auto image_size = tensor::build(INT32, {1, 2}, {transed_height, transed_width});

        fixed_image = Tensor::Pack({fixed_image, image_size});

        bench.input(0, fixed_image);

        bench.run();

        auto output = bench.output(0);
        auto boxlist = output.unpack();

        if (boxlist.size() != 4) {
            throw Exception("Boxlist length must be 4");
        }

        boxlist[0] = tensor::cast(INT32, boxlist[0]);
        boxlist[1] = tensor::cast(FLOAT32, boxlist[1]);
        boxlist[2] = tensor::cast(FLOAT32, boxlist[2]);
        boxlist[3] = tensor::cast(INT32, boxlist[3]);

        auto &inner_size = boxlist[0];
        auto &boxes = boxlist[1];
        auto &scores = boxlist[2];
        auto &labels = boxlist[3];
        auto N = boxes.size(0);

        auto x_scale = float(orignal_width) / inner_size.data<int32_t>(0);
        auto y_scale = float(orignal_height) / inner_size.data<int32_t>(1);

        std::vector<BindingBox> result;

        for (int i = 0; i < N; ++i) {
            auto score = scores.data<float>(i);
            if (score < threshold) continue;
            auto label = labels.data<int32_t>(i);
            auto box = boxes.data<XYXY>(i);

            box.x1 *= x_scale;
            box.y1 *= y_scale;
            box.x2 *= x_scale;
            box.y2 *= y_scale;

            result.emplace_back(box.x1, box.y1, box.x2, box.y2, score, label);
        }

        return std::move(result);
    }

    void load_categories(const std::string &path) {
        std::ifstream ifile(path);
        if (!ifile.is_open()) {
            std::cerr << "Can not open: " << path << std::endl;
            return;
        }
        this->categories.clear();
        std::string line;
        while (std::getline(ifile, line)) {
            categories.push_back(line);
        }
    }

    const std::string &category(int32_t label) const {
        return categories[label];
    }

private:
    Workbench bench = nullptr;
    ImageFilter filter = nullptr;
};

int main() {
    auto model = "/home/kier/OhMyTorch/object_detection/maskrcnn-benchmark/demo/output/grcnn/module.tsm";
    MaskRCNN mrcnn(model, "gpu", 0);
    mrcnn.threshold = 0.7f;

    mrcnn.load_categories("/home/kier/git/TensorStack/bin/categories.txt");

    auto cvimage = cv::imread("/home/kier/OhMyTorch/object_detection/maskrcnn-benchmark/demo/004545.png");
    auto boxes = mrcnn.run(cvimage);

    for (auto &box : boxes) {
        srand(box.label);

        auto r = rand() % 192 + 64;
        auto g = rand() % 192 + 64;
        auto b = rand() % 192 + 64;

        cv::rectangle(cvimage, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), CV_RGB(r, g, b), 2);
        cv::putText(cvimage, mrcnn.category(box.label), cv::Point(box.x1, box.y1), CV_FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(r - 32, g - 32, b - 32));
    }

    cv::imshow("Example", cvimage);
    cv::waitKey();

    return 0;
}
