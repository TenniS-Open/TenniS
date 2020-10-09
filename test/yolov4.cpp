//
// Created by sen on 8/26/20.
//
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "api/cpp/tensorstack.h"
#include "core/tensor.h"
#include <ctime>

//#define PYTHON_RESIZE

#ifdef PYTHON_RESIZE
#include "module/io/fstream.h"
#endif

struct BBox {
    float x1;
    float y1;
    float x2;
    float y2;

    float prob;
    int label;
};

static float IoU(const BBox &w1, const BBox &w2) {
    auto xOverlap = std::max<float>(0, std::min(w1.x2 - 1, w2.x2 - 1) - std::max(w1.x1, w2.x1) + 1);
    auto yOverlap = std::max<float>(0, std::min(w1.y2 - 1, w2.y2 - 1) - std::max(w1.y1, w2.y1) + 1);
    auto intersection = xOverlap * yOverlap;

    auto w1_width = w1.x2 - w1.x1;
    auto w1_height = w1.y2 - w1.y1;
    auto w2_width = w2.x2 - w2.x1;
    auto w2_height = w2.y2 - w2.y1;

    auto unio = w1_width * w1_height + w2_width * w2_height - intersection;
    return float(intersection) / unio;
}

static std::vector<BBox> NMS_sorted(std::vector<BBox> &winList, float threshold) {
    // sort BBox lists by prob in a descending order
    for (int i = 0; i < winList.size(); ++i) {
        for (int j = i + 1; j < winList.size(); ++j) {
            auto &lhs = winList[i];
            auto &rhs = winList[j];
            if (lhs.prob < rhs.prob) {
                std::swap(lhs, rhs);
            }
        }
    }

    if (winList.size() == 0)
        return winList;
    std::vector<bool> flag(winList.size(), false);
    for (size_t i = 0; i < winList.size(); i++) {
        if (flag[i]) continue;
        for (size_t j = i + 1; j < winList.size(); j++) {
            if (IoU(winList[i], winList[j]) > threshold) flag[j] = true;
        }
    }
    std::vector<BBox> ret;
    for (size_t i = 0; i < winList.size(); i++) {
        if (!flag[i]) ret.push_back(winList[i]);
    }
    return ret;
}

void drawRectangles(std::vector<BBox> &boxes, int &DEST, std::vector<std::string> &label, cv::Mat &cvimage) {
    std::ostringstream oss;
    BBox box;
    for (int i = 0; i < boxes.size(); ++i) {
        box = boxes[i];
#ifdef PYTHON_RESIZE
        // draw rectangle
        box.x1 *= float(cvimage.cols);
        box.y1 *= float(cvimage.rows);
        box.x2 *= float(cvimage.cols);
        box.y2 *= float(cvimage.rows);
#else
        // draw rectangle
        box.x1 *= float(DEST);
        box.y1 *= float(DEST);
        box.x2 *= float(DEST);
        box.y2 *= float(DEST);
#endif

        srand(int(i));
        auto r = rand() % 128 + 64;
        auto g = rand() % 128 + 64;
        auto b = rand() % 128 + 64;
        cv::rectangle(cvimage, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), CV_RGB(r, g, b), 2);

        oss.str("");
        if (label.empty()) {
            oss << int(box.label) << ": " << int(std::round(box.prob) * 100) << "%";
        } else {
            oss << label[int(box.label)] << ": " << int(std::round(box.prob * 100)) << "%";
        }
// cv::FONT_HERSHEY_DUPLEX
        std::cout << oss.str() << std::endl;
        cv::putText(cvimage, oss.str(), cv::Point(box.x1, box.y1 - 5), cv::FONT_HERSHEY_DUPLEX, 0.5,
                    CV_RGB(r - 32, g - 32, b - 32));
    }

}

int main() {
    using namespace ts::api;
    Device device("gpu", 0);

    auto start = std::chrono::system_clock::now();

//    cv::Mat cvimage = cv::imread("/home/sen/Downloads/yolo_tsm/yolov4_tsm//dog.jpg");
    cv::Mat cvimage = cv::imread("/home/sen/Downloads/horses.jpg");
    std::string model = "/home/sen/Workspace/gitlab/TenniS-frontend/tennis/test/convert.onnx.tsm";
    std::vector<std::string> label = {
            "person", "bicycle", "car", "motorbike",
            "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie",
            "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "pottedplant", "bed", "diningtable", "toilet",
            "tvmonitor", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush",
    };

    auto WIDTH = 608;  // the input of yolov4 is 608x608
    auto DEST = std::max(cvimage.rows, cvimage.cols);

    Workbench bench(device);
    bench.setup_context();
    bench.set_computing_thread_number(4);

    bench.setup(bench.compile(Module::Load(model)));

#ifdef PYTHON_RESIZE
    using ts::Tensor;
    using ts::FileStreamReader;
    ts::Tensor tTensor;
    ts::FileStreamReader ifile("/home/sen/Downloads/in_img.t");
    tTensor.externalize(ifile);
    ts::api::Tensor tensor = tensor::build(FLOAT32, {1, cvimage.channels(), WIDTH, WIDTH}, tTensor.data<float>());
#else
    ImageFilter filter(device);
    filter.channel_swap({2, 1, 0});
    filter.to_float();

    filter.scale(1 / 255.0);
    filter.letterbox(WIDTH, WIDTH, 0.5);    // darknet pre-processor
    filter.to_chw();
    bench.bind_filter(0, filter);

    Tensor tensor = tensor::build(UINT8, {1, cvimage.rows, cvimage.cols, cvimage.channels()}, cvimage.data);
#endif

    bench.input(0, tensor);

    bench.run();

    auto out = bench.output(0);  // [batch, num, 1, 4]
    auto confs = bench.output(1);  // [batch, num, num_classes]

    out = out.view(ts_InFlow(TS_HOST));
    confs = confs.view(ts_InFlow(TS_HOST));

    std::vector<BBox> boxLists;

    auto num = out.size(1);
    auto num_classes = confs.size(2);

    const float DETECT_THRESHOLD = 0.4;
    const float NMS_THRESHOLD = 0.6;

    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            auto prob = *(confs.data<float>() + i * num_classes + j);
            if (prob > DETECT_THRESHOLD) {
                BBox box;
                box.x1 = *(out.data<float>() + i * out.size(3));
                box.y1 = *(out.data<float>() + i * out.size(3) + 1);
                box.x2 = *(out.data<float>() + i * out.size(3) + 2);
                box.y2 = *(out.data<float>() + i * out.size(3) + 3);

                box.prob = prob;
                box.label = j;
                boxLists.push_back(box);
            }
        }
    }

    std::vector<BBox> out_boxes = NMS_sorted(boxLists, NMS_THRESHOLD);

    drawRectangles(out_boxes, DEST, label, cvimage);

    auto end = std::chrono::system_clock::now();
    auto dtime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << dtime.count() / 1000.0 << "ms" << std::endl;

    cv::namedWindow("YOLOv4", cv::WINDOW_AUTOSIZE);
    cv::imshow("YOLOv4", cvimage);
    cv::waitKey();

    return 0;
}