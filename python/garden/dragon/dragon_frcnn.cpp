//
// Created by kier on 19-4-28.
//

#include <runtime/workbench.h>
#include <runtime/image_filter.h>
#include <module/module.h>
#include <core/tensor_builder.h>
#include <board/hook.h>
#include <board/time_log.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <board/dump.h>

#include "bbox.h"

static inline void diff(const ts::Tensor &x, const ts::Tensor &y, float &max, float &avg) {
    auto float_x = ts::tensor::cast(ts::FLOAT32, x);
    auto float_y = ts::tensor::cast(ts::FLOAT32, y);
    auto count = std::min(x.count(), y.count());
    auto float_x_data = float_x.data<float>();
    auto float_y_data = float_y.data<float>();
    float local_max = 0;
    float local_sum = 0;
    for (int i = 0; i < count; ++i) {
        auto diff = std::fabs(float_x_data[i] - float_y_data[i]);
        local_sum += diff;
        if (local_max < diff) local_max = diff;
    }
    max = local_max;
    avg = local_sum / count;
}

void CopyCVMatToDataWithPad(
        const std::vector<cv::Mat*>&        mats,
        const int                           out_h,
        const int                           out_w,
        uint8_t*                            data) {
    static auto kBorderValue = cv::Scalar(127, 127, 127);

    auto* raw_data = data;
    cv::Mat padded_mat(out_h, out_w, CV_8UC3);

    for (int i = 0; i < mats.size(); i++) {
        TS_AUTO_CHECK_LE(mats[i]->rows, out_h);
        TS_AUTO_CHECK_LE(mats[i]->cols, out_w);
        int pad_bottom = out_h - mats[i]->rows;
        int pad_right = out_w - mats[i]->cols;
        cv::copyMakeBorder(*mats[i], padded_mat,
                           0, pad_bottom, 0, pad_right,
                           cv::BORDER_CONSTANT, kBorderValue);
        size_t n_bytes = sizeof(uint8_t) * padded_mat.rows *
                         padded_mat.cols * padded_mat.channels();
        memcpy(raw_data, padded_mat.data, n_bytes);
        raw_data += n_bytes;
    }
}

class Detector {
public:
    ts::Workbench::shared m_bench;

    static const int COARSEST_STRIDE = 32;

    bool vis_ = false;
    int scale_ = 600, max_size_ = 1000;
    float nms_ = 0.3f, thresh_ = 0.7f, im_scale_;

    ts::Tensor data;
    ts::Tensor im_info, rois, cls_prob, bbox_delta,data_im, conv1, pool1, C2, C5, rpn_cls_score_reshape, rpn_bbox_pred_reshape, rpn_bbox_pred_reshape_P2;
    ts::Tensor proposal_0, proposal_1, proposal_2,proposal_3;

    void set_vis(bool vis) { vis_ = vis; }
    void set_thresh(bool thresh) { thresh_ = thresh; }

    Detector(const std::string &model, const std::string &device=ts::CPU, int id = 0) {
        using namespace ts;
        ComputingDevice computing_device(device, id);

        auto module = Module::Load(model);
        auto bench = Workbench::Load(module, computing_device);
        bench->setup_context();

        auto filter = std::make_shared<ImageFilter>(device);
        filter->to_float();

        bench->bind_filter(0, filter);

        m_bench = bench;
    }

    void GetImageBlob(const cv::Mat& mat) {
        // Resize the image
        int im_size_min = std::min(mat.rows, mat.cols);
        int im_size_max = std::max(mat.rows, mat.cols);
        im_scale_ = float(scale_) / float(im_size_min);
        if (std::round(im_scale_ * im_size_max) > max_size_)
            im_scale_ = float(max_size_) / float(im_size_max);

        auto desired_h = mat.rows * im_scale_;
        auto desired_w = mat.cols * im_scale_;
        auto standard_h = int(std::ceil(desired_h / COARSEST_STRIDE) * COARSEST_STRIDE);
        auto standard_w = int(std::ceil(desired_w / COARSEST_STRIDE) * COARSEST_STRIDE);

        cv::Mat resize_mat;
        cv::resize(mat, resize_mat, cv::Size(0, 0),
                   im_scale_, im_scale_, cv::INTER_LINEAR);

        data = ts::Tensor(ts::UINT8, std::vector<int>({ 1, standard_h, standard_w, 3 }));
        CopyCVMatToDataWithPad(
               std:: vector<cv::Mat*>({ &resize_mat }),
                standard_h, standard_w, data.data<uint8_t >());

        im_info = ts::Tensor(ts::FLOAT32, std::vector<int>({ 1, 3 }));
        im_info.data<float>()[0] = (float)resize_mat.rows;
        im_info.data<float>()[1] = (float)resize_mat.cols;
        im_info.data<float>()[2] = im_scale_;
    }

    void RunGraph() {
        m_bench->input(0, data);
        m_bench->input(1, im_info);
        m_bench->run();
    }

    class Color {
    public:
        using self = Color;

        Color() = default;
        Color(uint8_t r, uint8_t g, uint8_t b)
                : r(r), g(g), b(b) {}

        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;

        static Color Random(uint8_t low, uint8_t high, int seed = 0) {
            if (seed) srand(seed);
            if (high < low) return self(low, low, low);
            auto r = uint8_t(low + rand() % (high - low + 1));
            auto g = uint8_t(low + rand() % (high - low + 1));
            auto b = uint8_t(low + rand() % (high - low + 1));
            return self(r, g, b);
        }

        operator cv::Scalar() const {
            return cv::Scalar(b, g, r);
        }
    };

    std::vector<BoundingBox> Run(const cv::Mat &image) {
        auto mat = image.clone();

        GetImageBlob(mat);

        RunGraph();

        std::vector<BoundingBox> result;

        rois = ts::tensor::cast(ts::FLOAT32, m_bench->output("rois"));              // (x, 5)
        cls_prob = ts::tensor::cast(ts::FLOAT32, m_bench->output("cls_prob"));      // (x, num_classes)
        bbox_delta = ts::tensor::cast(ts::FLOAT32, m_bench->output("bbox_delta"));  // (x, num_classes * 4)

        // Decode results
        auto* rois_data = rois.data<float>();
        auto* cls_prob_data = cls_prob.data<float>();
        auto* bbox_delta_data = bbox_delta.data<float>();
        auto num_classes = cls_prob.size(1);

#define CLS_PROB_OFFSET bbox_id * num_classes + cls
#define BBOX_DELTA_OFFSET (bbox_id * num_classes + cls) * 4

        std::vector< std::vector<BoundingBox> > detections(
                (size_t)cls_prob.size(1) - 1);

        for (int cls = 1; cls < num_classes; cls++) {
            std::vector<BoundingBox> keep;
            std::vector<int> keep_indices;
            // int count = 0;
            auto& cls_boxes = detections[cls - 1];
            for (int bbox_id = 0; bbox_id < cls_prob.size(0); bbox_id++) {
                float conf = cls_prob_data[CLS_PROB_OFFSET];

                if (conf < thresh_) continue;

                std::vector<float> roi({
                                               rois_data[bbox_id * 5 + 1] / im_scale_,
                                               rois_data[bbox_id * 5 + 2] / im_scale_,
                                               rois_data[bbox_id * 5 + 3] / im_scale_,
                                               rois_data[bbox_id * 5 + 4] / im_scale_});
                int64_t bbox_delta_offset = BBOX_DELTA_OFFSET;
                std::vector<float> delta({
                                                 bbox_delta_data[bbox_delta_offset] / 10.f,
                                                 bbox_delta_data[bbox_delta_offset + 1] / 10.f,
                                                 bbox_delta_data[bbox_delta_offset + 2] / 5.f,
                                                 bbox_delta_data[bbox_delta_offset + 3] / 5.f});
                BoundingBox box(roi, delta, conf, cls);
                box.Crop(mat.rows, mat.cols);

                keep.push_back(box);
            }
            sort(keep.begin(), keep.end());
            ApplyNMS(0.3, keep, keep_indices);

            for (auto i : keep_indices) cls_boxes.emplace_back(keep[i]);

            result.insert(result.end(), cls_boxes.begin(), cls_boxes.end());

            if (vis_) {
                auto title_color = Color::Random(128, 255, int(cls));
                auto box_color = Color(title_color.r - 128, title_color.g - 128, title_color.b - 128);
                for (const auto& bbox : cls_boxes) {
                    cv::rectangle(mat,
                                  cvPoint(bbox.ix1(), bbox.iy1()),
                                  cvPoint(bbox.ix2(), bbox.iy2()),
                                  box_color, 2, 1, 0);
                    cv::putText(mat, std::to_string(int(bbox.cls()))+": "+std::to_string(bbox.conf()),
                                cvPoint(bbox.ix1(), bbox.iy1() - 10), cv::FONT_HERSHEY_DUPLEX, 0.5, title_color, 1);
                }
            }

        }

        if (vis_) {
            /*
            for (int cls = 0; cls < detections.size(); cls++) {
                for (const auto& bbox : detections[cls]) {
                    cv::rectangle(mat,
                                  cvPoint(bbox.ix1(), bbox.iy1()),
                                  cvPoint(bbox.ix2(), bbox.iy2()),
                                  cv::Scalar(18, 127, 15), 2, 1, 0);
                    cv::putText(mat, std::to_string(cls),
                                cvPoint(bbox.ix1(), bbox.iy1()), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 3), 2);
                }
            }
            */
            cv::imshow("FasterRCNN", mat);
            cv::waitKey(0);
        }

        return result;
    }

};

int main() {
    using namespace ts;

    std::string model = "/home/kier/git/TensorStack/python/test/frcnn_from_dragon.tsm";

    Detector detector(model, GPU, 0);
    detector.set_vis(true);

    auto image = cv::imread("icebox/000471.jpg");
    cv::imwrite("000471.png", image);

    std::cout << "Image: [" << image.cols << ", " << image.rows << ", " << image.channels() << "]" << std::endl;

    auto bboxs = detector.Run(image);
    for (auto &box : bboxs) {
        std::cout << box.cls() << ": ["
                  << box.ix1() << ", " << box.iy1() << ", " << box.ix2() << ", " << box.iy2()
                  << "] " << box.conf() << std::endl;
    }

    /*
    detector.m_bench->do_profile(true);
    {
        int N = 10;
        TimeLog _log(LOG_INFO, "Every takes: ");
        _log.denominator(N);
        for (auto i = 0; i < N; ++i) {
            detector.Run(image);
        }
    }
    detector.m_bench->profiler().log(std::cout);
    */

    return 0;
}