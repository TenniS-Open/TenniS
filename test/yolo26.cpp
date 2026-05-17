//
// Created by Levalup.
// L.eval: Let programmer get rid of only work jobs.
//

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "api/cpp/tennis.h"

struct Preprocessed {
    float scale;
    float shift_x;
    float shift_y;
};

struct Rect {
    float x1, y1, x2, y2;
    float score;
    float clazz;

    operator cv::Rect() const {
        return cv::Rect(x1, y1, x2 - x1, y2 - y1);
    }

    Rect &operator*=(float v) {
        x1 *= v;
        y1 *= v;
        x2 *= v;
        y2 *= v;
        return *this;
    }

    Rect &operator-=(Preprocessed p) {
        x1 = (x1 - p.shift_x) / p.scale;
        y1 = (y1 - p.shift_y) / p.scale;
        x2 = (x2 - p.shift_x) / p.scale;
        y2 = (y2 - p.shift_y) / p.scale;
        return *this;
    }
};

struct Point {
    float x, y;
    float score;

    operator cv::Point() const {
        return cv::Point(x, y);
    }

    Point &operator*=(float v) {
        x *= v;
        y *= v;
        return *this;
    }

    Point &operator-=(Preprocessed p) {
        x = (x - p.shift_x) / p.scale;
        y = (y - p.shift_y) / p.scale;
        return *this;
    }
};

struct Target {
    Rect rect;
    std::vector<Point> points;

    Target &operator*=(float v) {
        rect *= v;
        for (auto &point : points) {
            point *= v;
        }
        return *this;
    }

    Target &operator-=(Preprocessed p) {
        rect -= p;
        for (auto &point : points) {
            point -= p;
        }
        return *this;
    }
};

Target decode(const float *data, int size) {
    auto data_rect = reinterpret_cast<const Rect *>(data);
    auto data_points = reinterpret_cast<const Point *>(data + 6);

    Target target;
    target.rect = *data_rect;
    auto n = (size - 6) / 3;
    target.points.resize(n);
    for (int i = 0; i < n; i++) {
        target.points[i] = data_points[i];
    }

    return target;
}

Preprocessed letterbox(int net_width, int net_height, int image_width, int image_height) {
    auto scale_width = static_cast<float>(net_width) / static_cast<float>(image_width);
    auto scale_height = static_cast<float>(net_height) / static_cast<float>(image_height);

    Preprocessed p{};
    if (scale_width < scale_height) {
        p.scale = scale_width;
        p.shift_x = 0;
        p.shift_y = (static_cast<float>(net_height) - static_cast<float>(image_height) * scale_height) / 2.0f;
    } else {
        p.scale = scale_height;
        p.shift_y = 0;
        p.shift_x = (static_cast<float>(net_width) - static_cast<float>(image_width) * scale_width) / 2.0f;
    }

    return p;
}

int main() {
    using namespace ts::api;
    Device device("cpu", 0);

    cv::Mat cvimage = cv::imread("/Users/levalup/Downloads/linger.jpeg");
    std::string model = "/Users/levalup/Workspace/ProjectDev/C2 NewFace/fd/0513/best.tsm";

    float threshold = 0.5;

    auto NET = 640;  // the input of yolo26 is 640x640
    auto BOLD = 10;

    auto preprocessed = letterbox(NET, NET, cvimage.cols, cvimage.rows);

    Workbench bench(device);
    bench.setup_context();
    bench.set_computing_thread_number(4);

    bench.setup(bench.compile(Module::Load(model)));

    ImageFilter filter(device);
    filter.channel_swap({2, 1, 0});
    filter.to_float();

    filter.scale(1 / 255.0);
    filter.letterbox(NET, NET, 0.5);    // darknet pre-processor
    filter.to_chw();
    bench.bind_filter(0, filter);

    Tensor tensor = tensor::build(UINT8, {1, cvimage.rows, cvimage.cols, cvimage.channels()}, cvimage.data);

    bench.input(0, tensor);

    bench.run();

    auto out = bench.output(0);

    out = out.view(TS_HOST);

    auto n = out.size(0) * out.size(1);
    auto k = out.size(2);
    auto data = out.data<float>();

    std::vector<Target> targets;

    for (int i = 0; i < n; i++) {
        auto score = data[4];
        if (score < threshold) continue;

        auto target = decode(data, k);
        target -= preprocessed;
        targets.push_back(target);

        data += k;
    }

    std::cout << "Detected " << targets.size() << " target(s)" << std::endl;

    for (auto &target : targets) {
        cv::rectangle(cvimage, target.rect, CV_RGB(0, 255, 0), BOLD);
        for (auto &point : target.points) {
            cv::circle(cvimage, point, BOLD, CV_RGB(255, 0, 0), -1);
        }
    }

    cv::imshow("Image", cvimage);
    cv::waitKey(-1);

    return 0;
}
