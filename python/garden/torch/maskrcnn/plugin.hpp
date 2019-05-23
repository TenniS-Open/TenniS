//
// Created by kier on 19-5-16.
//

#ifndef TENSORSTACK_PLUGIN_HPP
#define TENSORSTACK_PLUGIN_HPP


#include <cmath>
#include <algorithm>
#include <sstream>

#include "api/cpp/tensorstack.h"
#include "api/cpp/operator.h"

#include "roi_align.hpp"

using namespace ts;

class AnchorGenerator : public api::Operator {
private:
    std::vector<int> strides;
    std::vector<api::Tensor> cell_anchors;

public:
    void init(const api::OperatorParams &params, ts_OperatorContext *) override {
        auto param_strides = params.get("strides");
        auto param_cell_anchors = params.get("cell_anchors");
        if (param_strides == nullptr || param_cell_anchors == nullptr) {
            TS_THROW("Init: strides and cell_anchors must be set");
        }
        param_strides = param_strides.cast(api::INT32);
        auto unpacked_cell_anchors = param_cell_anchors.unpack();
        for (auto &base_anchros : unpacked_cell_anchors) {
            cell_anchors.emplace_back(base_anchros.cast(api::FLOAT32));
        }
        auto count = param_strides.count();
        strides = std::vector<int>(param_strides.data<int32_t>(), param_strides.data<int32_t>() + count);

        if (strides.size() != cell_anchors.size()) {
            TS_THROW("Init: strides and cell_anchors must be save length, got " +
                     std::to_string(strides.size()) + " vs. " + std::to_string(cell_anchors.size()));
        }

    }

    std::vector<std::pair<api::DTYPE, api::Shape>> infer(const std::vector<api::Tensor> &args, ts_OperatorContext *) override {
        TS_THROW("Not implement");
        return std::vector<std::pair<api::DTYPE, api::Shape>>();
    }

    template <typename T>
    static std::vector<T> torch_arange(int begin, int end, int step) {
        std::vector<T> range;
        int value = begin;
        while (value < end) {
            range.emplace_back(value);
            value += step;
        }
        return std::move(range);
    }

    template <typename T>
    std::tuple<std::vector<T>, std::vector<T>> torch_meshgrid(const std::vector<T> &dim0, const std::vector<T> &dim1) {
        std::tuple<std::vector<T>, std::vector<T>> grid;
        auto &grid_dim0 = std::get<0>(grid);
        auto &grid_dim1 = std::get<1>(grid);
        auto dim0_size = dim0.size();
        auto dim1_size = dim1.size();
        auto count = dim0_size * dim1_size;
        grid_dim0.reserve(count);
        grid_dim1.reserve(count);
        // set grid_dim0
        for (size_t row = 0; row < dim0_size; ++row) {
            grid_dim0.insert(grid_dim0.end(), dim1_size, dim0[row]);
        }
        // set grad_dim1
        for (size_t row = 0; row < dim0_size; ++row) {
            grid_dim1.insert(grid_dim1.end(), dim1.begin(), dim1.end());
        }
        return std::move(grid);
    }

    template <typename T>
    std::vector<T> torch_stack_dim_1(
            const std::vector<T> &x0,
            const std::vector<T> &x1,
            const std::vector<T> &x2,
            const std::vector<T> &x3) {
        std::vector<T> stack;
        stack.resize(x0.size() * 4);
        auto data = stack.data();
        auto x0_data = x0.data();
        auto x1_data = x1.data();
        auto x2_data = x2.data();
        auto x3_data = x3.data();
        auto width = x0.size();
        for (size_t i = 0; i < width; ++i) {
            *data++ = *x0_data++;
            *data++ = *x1_data++;
            *data++ = *x2_data++;
            *data++ = *x3_data++;
        }
        return std::move(stack);
    }

    std::vector<std::vector<float>> grid_anchors(const std::vector<int32_t> &grid_sizes) {
        auto grid_sizes_length = grid_sizes.size() / 2;
        if (grid_sizes_length != strides.size() || strides.size() != cell_anchors.size()) {
            TS_THROW("Init: grid_sizes, strides and cell_anchors must be save length, got (" +
                     std::to_string(grid_sizes_length) + ", " +
                     std::to_string(strides.size()) + ", " +
                     std::to_string(cell_anchors.size()) + ")");
        }

        std::vector<std::vector<float>> anchors;

        const auto len = grid_sizes_length;
        for (size_t i = 0; i < len; ++i) {
            auto &size = grid_sizes[i * 2];
            auto &stride = strides[i];
            auto &base_anchors = cell_anchors[i];

            int grid_height = size;
            int grid_width = *(&size + 1);

            auto shifts_x = torch_arange<float>(0, grid_width * stride, stride);
            auto shifts_y = torch_arange<float>(0, grid_height * stride, stride);

            auto _shift = torch_meshgrid(shifts_y, shifts_x);
            auto &shift_y = std::get<0>(_shift);
            auto &shift_x = std::get<1>(_shift);

            auto shifts = torch_stack_dim_1(shift_x, shift_y, shift_x, shift_y);

            std::vector<float> final_anchors;   // shape[m, n, 4]
            auto m = shifts.size() / 4;
            auto n = base_anchors.count() / 4;
            final_anchors.resize(m * n * 4);

            auto final_anchors_data = final_anchors.data();
            auto shifts_data = shifts.data();
            auto base_anchors_data = base_anchors.data<float>();

            auto shifts_data_k = shifts_data;
            for (int k = 0; k < m; ++k) {
                auto base_anchors_data_j = base_anchors_data;
                for (int j = 0; j < n; ++j) {
                    auto shifts_data_i_j = shifts_data_k;
                    *final_anchors_data++ = *shifts_data_i_j++ + *base_anchors_data_j++;
                    *final_anchors_data++ = *shifts_data_i_j++ + *base_anchors_data_j++;
                    *final_anchors_data++ = *shifts_data_i_j++ + *base_anchors_data_j++;
                    *final_anchors_data++ = *shifts_data_i_j++ + *base_anchors_data_j++;
                }
                shifts_data_k += 4;
            }

            anchors.emplace_back(std::move(final_anchors));
        }

        return std::move(anchors);
    }

    std::vector<api::Tensor> run(const std::vector<api::Tensor> &args, ts_OperatorContext *) override {
        if (args.size() != 2) TS_THROW("AnchorGenerator: input number must be 2");

        auto images = args[0].view(api::Tensor::InFlow::HOST);
        auto features = args[1].view(api::Tensor::InFlow::HOST);

        if (images.dtype() != api::FLOAT32 || features.dtype() != api::FLOAT32) {
            TS_THROW("AnchorGenerator: input images and features only support float32");
        }

        int number = images.size(0);
        if (number != 1) {
            TS_THROW("AnchorGenerator: batch size must be 1");
        }

        std::vector<int32_t> image_sizes;
        if (images.packed()) {
            auto size_field = images.field(1).cast(api::INT32);
            if (number != size_field.size(0) || size_field.size(1) != 2) {
                TS_THROW("AnchorGenerator: image_sizes'shape must be [batch_size, 2]");
            }
            image_sizes.resize(number * 2);
            std::memcpy(image_sizes.data(), size_field.data(), number * 2 * 4);
        } else {
            image_sizes.reserve(number * 2);
            auto image_height = images.size(2);
            auto image_width = images.size(3);
            for (int n = 0; n < number; ++n) {
                image_sizes.emplace_back(image_height);
                image_sizes.emplace_back(image_width);
            }
        }

        auto feature_maps = features.unpack();

        // int grid_height = feature_maps[0].size(2);
        // int grid_width = feature_maps[0].size(3);

        std::vector<int32_t> grid_sizes;
        for (auto &feature_map : feature_maps) {
            grid_sizes.emplace_back(feature_map.size(2));
            grid_sizes.emplace_back(feature_map.size(3));
        }

        std::vector<std::vector<float>> anchors_over_all_feature_maps = grid_anchors(grid_sizes);

        std::vector<api::Tensor> anchors;   // Repeating [ListAnchorNumber, [image_width, image_height], Anchor[-1, 4], ...]

        // write ListAnchorNumber
        std::vector<int32_t> list_anchor_number(number, int32_t(anchors_over_all_feature_maps.size()));
        anchors.emplace_back(api::tensor::build<int32_t>(api::INT32, number, list_anchor_number.data()));

        for (int n = 0; n < number; ++n) {
            // auto i = n;
            auto image_height = image_sizes[2 *  n];
            auto image_width = image_sizes[2 *  n + 1];

            for (auto &anchors_per_feature_map : anchors_over_all_feature_maps) {
                // write [image_width, image_height]
                anchors.emplace_back(api::tensor::build<int32_t>(api::INT32, {2}, {image_width, image_height}));
                // write Anchor[-1, 4]
                anchors.emplace_back(api::tensor::build<float>(api::FLOAT32,
                                                               {int32_t(anchors_per_feature_map.size()) / 4, 4},
                                                               anchors_per_feature_map.data()));
            }
        }

        return std::move(anchors);
    }
};

/**
:param name:
:param anchors: Repeating [ListAnchorNumber, [image_width, image_height], Anchor[-1, 4], ...]
:param objectness: PackedTensor for each scale objectness
:param rpn_box_regression: PackedTensor for each scale
:return: Repeating [[image_width, image_height], Boxes[-1, 4], Scores]
 */
class BoxSelector : public api::Operator {
private:
    int pre_nms_top_n = 100;

    std::vector<float> weights;
    std::vector<float> inverse_weights;
    float bbox_xform_clip = std::log(1000.0f / 16);

    int min_size;
    float nms_thresh;
    int post_nms_top_n;

    int fpn_post_nms_top_n;
public:
    void init(const api::OperatorParams &params, ts_OperatorContext *) override {
        this->pre_nms_top_n = api::tensor::to_int(params.get("pre_nms_top_n"));
        auto weights_tensor = api::tensor::cast(api::FLOAT32, params.get("weights"));
        auto weights_data = weights_tensor.data<float>();
        auto weights_count = weights_tensor.count();
        this->weights = std::vector<float>(weights_data, weights_data + weights_count);
        this->bbox_xform_clip = api::tensor::to_float(params.get("bbox_xform_clip"));

        this->inverse_weights = this->weights;
        for (auto &weight : this->inverse_weights) {
            weight = 1.0f / weight;
        }

        this->min_size = api::tensor::to_int(params.get("min_size"));
        this->nms_thresh = api::tensor::to_float(params.get("nms_thresh"));
        this->post_nms_top_n = api::tensor::to_int(params.get("post_nms_top_n"));

        this->fpn_post_nms_top_n = api::tensor::to_int(params.get("fpn_post_nms_top_n"));
    }

    std::vector<std::pair<api::DTYPE, api::Shape>> infer(const std::vector<api::Tensor> &args, ts_OperatorContext *) override {
        TS_THROW("Not implement");
        return std::vector<std::pair<api::DTYPE, api::Shape>>();
    }

    static std::vector<std::vector<api::Tensor>> unpack_anchor(const std::vector<api::Tensor> &packed_anchors) {
        auto &list_anchor_number = packed_anchors[0];
        auto num = list_anchor_number.count();
        auto data = list_anchor_number.data<int32_t>();

        std::vector<std::vector<api::Tensor>> unpacked_anchors;

        auto begin = packed_anchors.begin() + 1;
        for (int i = 0; i < num; ++i) {
            auto size = data[i];
            auto end = begin + size * 2;

            unpacked_anchors.emplace_back(std::vector<api::Tensor>(begin, end));
            begin = end;
        }

        return std::move(unpacked_anchors);
    }

    template <typename T>
    static std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> &x, int width = 2) {
        size_t M = x[0].size() / width;
        std::vector<std::vector<T>> y(M);

        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < x.size(); ++n) {
                auto shift = width * m;
                y[m].insert(y[m].end(), x[n].begin() + shift, x[n].begin() + shift + width);
            }
        }

        return std::move(y);
    }

    /**
     * 2D topk
     * @param source
     * @param k
     * @param dim
     * @return
     */
    static std::tuple<api::Tensor, api::Tensor> torch_2d_topk_dim1(const api::Tensor &source, int k) {
        auto dim0 = source.size(0);
        auto dim1 = source.size(1);

        k = std::min<int>(k, dim1);

        api::Tensor value(api::Tensor::InFlow::HOST, api::FLOAT32, {dim0, k});
        api::Tensor index(api::Tensor::InFlow::HOST, api::INT32, {dim0, k});

        auto source_cpu = source.view(api::Tensor::InFlow::HOST);

        for (int i = 0; i < dim0; ++i) {
            auto value_data = value.data<float>() + i * k;
            int32_t *index_data = index.data<int32_t>() + i * k;
            int32_t index_size = 0;
            auto source_data = source_cpu.data<float>() + i * dim1;

            auto index_compare = [&](int32_t lhs, int32_t rhs) -> bool {
                return source_data[lhs] > source_data[rhs];
            };

            for (int32_t n = 0; n < dim1; ++n) {
                if (index_size < k) {
                    index_data[index_size++] = n;
                    std::push_heap(index_data, index_data + index_size, index_compare);
                } else {
                    std::pop_heap(index_data, index_data + index_size, index_compare);
                    index_data[k - 1] = n;
                    std::push_heap(index_data, index_data + index_size, index_compare);
                }
            }

            std::sort_heap(index_data, index_data + index_size, index_compare);

            for (int32_t n = 0; n < k; ++n) {
                value_data[n] = source_data[index_data[n]];
            }
        }

        return std::make_tuple(value, index);
    }

    /**
     * 2D topk
     * @param source
     * @param k
     * @param dim
     * @return
     */
    static std::tuple<api::Tensor, api::Tensor> torch_1d_topk_dim0(const api::Tensor &source, int k) {
        auto dim0 = source.size(0);
        // auto dim1 = source.size(1);

        k = std::min<int>(k, dim0);

        api::Tensor value(api::Tensor::InFlow::HOST, api::FLOAT32, {k});
        api::Tensor index(api::Tensor::InFlow::HOST, api::INT32, {k});

        auto source_cpu = source.view(api::Tensor::InFlow::HOST);

        auto value_data = value.data<float>();
        auto *index_data = index.data<int32_t>();
        int32_t index_size = 0;
        auto source_data = source_cpu.data<float>();

        auto index_compare = [&](int32_t lhs, int32_t rhs) -> bool {
            return source_data[lhs] > source_data[rhs];
        };

        for (int32_t n = 0; n < dim0; ++n) {
            if (index_size < k) {
                index_data[index_size++] = n;
                std::push_heap(index_data, index_data + index_size, index_compare);
            } else {
                std::pop_heap(index_data, index_data + index_size, index_compare);
                index_data[k - 1] = n;
                std::push_heap(index_data, index_data + index_size, index_compare);
            }
        }

        std::sort_heap(index_data, index_data + index_size, index_compare);

        for (int32_t n = 0; n < k; ++n) {
            value_data[n] = source_data[index_data[n]];
        }

        return std::make_tuple(value, index);
    }

    api::Tensor boxcoder_decode(const api::Tensor &rel_codes, const api::Tensor &boxes) {
        /*"""
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """*/

        api::Tensor pred_boxes(api::Tensor::InFlow::HOST, api::FLOAT32, rel_codes.sizes());
        auto rel_codes_width = rel_codes.size(1);
        auto boxes_width = boxes.size(1);

        auto N = rel_codes.size(0);
        if (boxes.size(0) != N) {
            TS_THROW("BBox decoding failed.");
        }

        const auto TO_REMOVE = 1;

        for (int i = 0; i < N; ++i) {
            auto rel_codes_data = rel_codes.data<float>() + i * rel_codes_width;
            auto boxes_data = boxes.data<float>() + i * boxes_width;
            auto pred_boxes_data = pred_boxes.data<float>() + i * rel_codes_width;

            auto width = boxes_data[2] - boxes_data[0] + TO_REMOVE;
            auto height = boxes_data[3] - boxes_data[1] + TO_REMOVE;

            auto ctr_x = boxes_data[0] + 0.5f * width;
            auto ctr_y = boxes_data[1] + 0.5f * height;

            auto dx = rel_codes_data[0] * inverse_weights[0];
            auto dy = rel_codes_data[1] * inverse_weights[1];
            auto dw = rel_codes_data[2] * inverse_weights[2];
            auto dh = rel_codes_data[3] * inverse_weights[3];

            if (dw > bbox_xform_clip) dw = bbox_xform_clip;
            if (dh > bbox_xform_clip) dh = bbox_xform_clip;

            auto pred_ctr_x = dx * width + ctr_x;
            auto pred_ctr_y = dy * height + ctr_y;
            auto pred_w = std::exp(dw) * width;
            auto pred_h = std::exp(dh) * height;

            // x1
            pred_boxes_data[0] = pred_ctr_x - 0.5f * pred_w;
            // y1
            pred_boxes_data[1] = pred_ctr_y - 0.5f * pred_h;
            // x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
            pred_boxes_data[2] = pred_ctr_x + 0.5f * pred_w - 1;
            // y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
            pred_boxes_data[3] = pred_ctr_y + 0.5f * pred_h - 1;
        }

        return std::move(pred_boxes);
    }

    template <typename T>
    static inline T clamp(T x, T min, T max) {
        return std::max(min, std::min(max, x));
    }

    static float IoU(const float *w1, const float *w2) {
        auto xOverlap = std::max<float>(0, std::min(w1[2] - 1, w2[2] - 1) - std::max(w1[0], w2[0]) + 1);
        auto yOverlap = std::max<float>(0, std::min(w1[3] - 1, w2[3] - 1) - std::max(w1[1], w2[1]) + 1);
        auto intersection = xOverlap * yOverlap;

        auto w1_width = w1[2] - w1[0];
        auto w1_height = w1[3] - w1[1];
        auto w2_width = w2[2] - w2[0];
        auto w2_height = w2[3] - w2[1];

        auto unio = w1_width * w1_height + w2_width * w2_height - intersection;
        return float(intersection) / unio;
    }

    /**
     *
     * @param size
     * @param width
     * @param height
     * @param boxes
     * @param scores
     * @param min_size
     * @param nms_thresh
     * @param max_proposals
     * @return box, score
     */
    static std::vector<api::Tensor> nms(int size,
                                        int width, int height,
                                        api::Tensor boxes, const api::Tensor &scores, int min_size, float nms_thresh, int max_proposals) {
        std::vector<bool> flag(size, false);    // remove flag
        static const auto WIDTH = 4;
        static const auto TO_REMOVE = 1;

        auto boxes_data = boxes.data<float>();
        // auto scores_data = scores.data<float>();

        for (int i = 0; i < size; ++i) {
            auto box = boxes_data + i * WIDTH;
            // auto score = scores + i;
            box[0] = clamp<float>(box[0], 0, width - TO_REMOVE);
            box[1] = clamp<float>(box[1], 0, height - TO_REMOVE);
            box[2] = clamp<float>(box[2], 0, width - TO_REMOVE);
            box[3] = clamp<float>(box[3], 0, height - TO_REMOVE);
            if ((box[2] - box[0]) < min_size || (box[3] - box[1]) < min_size) {
                flag[i] = true;
            }
        }

        int count = 0;
        for (int i = 0; i < size; i++) {
            if (flag[i]) continue;
            else ++count;
            for (int j = i + 1; j < size; j++) {
                if (flag[j]) continue;
                if (IoU(&boxes_data[i * WIDTH], &boxes_data[j * WIDTH]) > nms_thresh) flag[j] = true;
            }
        }

        std::vector<int32_t> kept_ind;
        kept_ind.reserve(max_proposals);

        int kept_size = 0;

        for (int i = 0; i < size; i++) {
            if (kept_size >= max_proposals) break;

            if (flag[i]) continue;
            kept_ind.push_back(i);

            ++kept_size;
        }

        auto kept_ind_tensor = api::tensor::build(api::INT32, kept_ind);
        auto kept_boxes = api::intime::gather(boxes, kept_ind_tensor, 0);
        auto kept_scores = api::intime::gather(scores, kept_ind_tensor, 0);

        return {kept_boxes, kept_scores};
    }

    /**
     *
     * @param anchors list[BoxList], BoxList means [image_size, anchors[-1, 4]]
     * @param objectness tensor of size N, A, H, W
     * @param box_regression tensor of size N, A * 4, H, W
     * @return Repeating [image_size, anchors[N, 4], score[N]]
     */
    std::vector<api::Tensor> forward_for_single_feature_map(
            const std::vector<api::Tensor> &anchors,
            api::Tensor objectness, api::Tensor box_regression, int level=0) {
        auto NAHW = ts_Tensor_shape(objectness.get_raw());
        auto N = NAHW[0];
        auto A = NAHW[1];
        auto H = NAHW[2];
        auto W = NAHW[3];

        if (N != 1) {
            TS_THROW("BoxSelector: batch size must be 1");
        }

        objectness = api::intime::transpose(objectness, {0, 2, 3, 1}).reshape({N, -1});
        objectness = api::intime::sigmoid(objectness);

        box_regression = api::intime::transpose(box_regression.reshape({N, -1, 4, H, W}), {0, 3, 4, 1, 2});
        box_regression = box_regression.reshape({N, -1, 4});

        auto num_anchors = A * H * W;
        auto min_pre_nms_top_n = std::min<int32_t>(this->pre_nms_top_n, num_anchors);

        api::Tensor topk_idx = nullptr;

        std::tie(objectness, topk_idx) = torch_2d_topk_dim1(objectness, min_pre_nms_top_n);


        topk_idx = topk_idx.reshape({topk_idx.count()});

        box_regression = api::intime::gather(box_regression, topk_idx, 1);  // TODO: this is not support N > 1

        std::vector<api::Tensor> list_anchors;
        auto list_anchors_size = anchors.size() / 2;
        for (size_t i = 0; i < list_anchors_size; ++i) {
            list_anchors.push_back(anchors[i * 2 + 1]);
        }

        auto concat_anchors = api::intime::concat(list_anchors, 0).reshape({N, -1, 4});
        concat_anchors = api::intime::gather(concat_anchors, topk_idx, 1);  // TODO: this is not support N > 1

        // docode boxes
        api::Tensor proposals = boxcoder_decode(
                box_regression.view(api::Tensor::InFlow::HOST).reshape({-1, 4}),
                concat_anchors.view(api::Tensor::InFlow::HOST).reshape({-1, 4}));

        proposals = proposals.reshape({N, -1, 4});

        // auto proposals_width = proposals.size(1) * proposals.size(2);
        auto objectness_width = objectness.size(1);

        std::vector<api::Tensor> result;

        for (int n = 0; n < N; ++n) {
            // do clip_to_image, remove_small_boxes and nms
            auto im_shape = anchors[n * 2];
            auto width = im_shape.data<int>(0);
            auto height = im_shape.data<int>(1);
            // list [box, score]
            auto boxlist = nms(objectness_width,
                               width, height, proposals.slice(n), objectness.slice(n),
                               min_size, nms_thresh, post_nms_top_n);
            // push im_shape, box, score
            result.emplace_back(im_shape);
            result.emplace_back(boxlist[0]);
            result.emplace_back(boxlist[1]);
        }

        return std::move(result);
    }

    static std::vector<api::Tensor> cat_boxlist(const std::vector<api::Tensor> &boxes, int width = 3, int ind = 1) {
        std::vector<api::Tensor> boxlist(width, nullptr);
        int i = 0;
        for (; i < ind; ++i) {
            boxlist[i] = boxes[i];
        }
        auto boxes_size = int(boxes.size());
        auto num = boxes_size / width;
        std::vector<api::Tensor> fields;
        fields.reserve(num);
        for (; i < width; ++i) {
            fields.clear();
            for (int j = i; j < boxes_size; j += width) {
                fields.emplace_back(boxes[j]);
            }
            boxlist[i] = api::intime::concat(fields, 0);
        }
        return boxlist;
    }

    static std::vector<std::vector<api::Tensor>> cat_boxlist(const std::vector<std::vector<api::Tensor>> &boxes, int width = 3, int ind = 1) {
        std::vector<std::vector<api::Tensor>> boxlist;
        for (auto &sublist : boxes) {
            boxlist.emplace_back(cat_boxlist(sublist, width, ind));
        }
        return boxlist;
    }

    static std::vector<api::Tensor> flatten(const std::vector<std::vector<api::Tensor>> &x) {
        std::vector<api::Tensor> y;
        for (auto &item : x) {
            y.insert(y.end(), item.begin(), item.end());
        }
        return std::move(y);
    }

    // input output list[list[im'size, boxes, scores]]
    std::vector<std::vector<api::Tensor>> select_over_all_levels(const std::vector<std::vector<api::Tensor>> &boxlists) {
        std::vector<std::vector<api::Tensor>> selected;
        for (auto &boxlist : boxlists) {
            api::Tensor scores = nullptr;
            api::Tensor index = nullptr;
            std::tie(scores, index) = torch_1d_topk_dim0(boxlist[2].view(api::Tensor::InFlow::HOST),
                                                         std::min(this->fpn_post_nms_top_n, boxlist[2].size(0)));
            api::Tensor boxes = api::intime::gather(boxlist[1], index, 0);
            selected.emplace_back(std::vector<api::Tensor>({boxlist[0], boxes, scores}));
        }
        return selected;
    }

    std::vector<api::Tensor> run(const std::vector<api::Tensor> &args, ts_OperatorContext *) override {
        if (args.size() != 3) TS_THROW("BoxSelector: input number must be 3");
        auto packed_anchors = args[0].view(api::Tensor::InFlow::HOST);
        auto packed_objectness = args[1].view(api::Tensor::InFlow::HOST);
        auto packed_box_regression = args[2].view(api::Tensor::InFlow::HOST);

        auto anchors = transpose(unpack_anchor(packed_anchors.unpack()));
        auto objectness = packed_objectness.unpack();
        auto box_regression = packed_box_regression.unpack();

        auto num_levels = objectness.size();

        if (anchors.size() != objectness.size() || objectness.size() != box_regression.size()) {
            TS_THROW("Init: anchors, objectness and box_regression must be same length, got (" +
                     std::to_string(anchors.size()) + ", " +
                     std::to_string(objectness.size()) + ", " +
                     std::to_string(box_regression.size()) + ")");
        }

        std::vector<std::vector<api::Tensor>> sampled_boxes;
        for (size_t i = 0; i <  num_levels; ++i) {
            // if (i != 4) continue;
            auto &a = anchors[i];
            auto &o = objectness[i];
            auto &b = box_regression[i];
            sampled_boxes.emplace_back(forward_for_single_feature_map(a, o, b, i));
        }

        auto boxlists = transpose(sampled_boxes, 3);

        boxlists = cat_boxlist(boxlists);

        if (num_levels > 1) {
            // select_over_all_levels
            boxlists = select_over_all_levels(boxlists);
        }

        return flatten(boxlists);
    }
};

class ROIAlign {
public:
    using self = ROIAlign;
    using shared = std::shared_ptr<self>;
private:
    std::vector<int32_t> output_size;
    float spatial_scale;
    int sampling_ratio;
public:
    ROIAlign(const std::vector<int32_t> &output_size, float spatial_scale, int sampling_ratio)
        : output_size(output_size), spatial_scale(spatial_scale), sampling_ratio(sampling_ratio) {}

    api::Tensor forward(const api::Tensor &x, const api::Tensor &rois) {
        return api::ROIAlign_forward_cpu(x.view(api::Tensor::InFlow::HOST), rois.view(api::Tensor::InFlow::HOST),
                                         spatial_scale, output_size[0], output_size[1], sampling_ratio);
    }
};

class LevelMapper {
public:
    using self = LevelMapper;
    using shared = std::shared_ptr<self>;

    int k_min;
    int k_max;
    int s0;
    int lvl0;
    float eps;
public:
    LevelMapper(int k_min, int k_max, int canonical_scale=224, int canonical_level=4, float eps=1e-6)
        : k_min(k_min), k_max(k_max), s0(canonical_scale), lvl0(canonical_level), eps(eps) {}

    template <typename T>
    static inline T clamp(T x, T min, T max) {
        return std::max(min, std::min(max, x));
    }

    api::Tensor forward(const std::vector<api::Tensor> &boxes, int width = 3, int ind = 1) {
        auto size = int(boxes.size());
        std::vector<api::Tensor> bbox;
        auto target_size = 0;
        for (auto i = ind; i < size; i += width) {
            bbox.emplace_back(boxes[i]);
            target_size += boxes[i].size(0);
        }

        api::Tensor target(api::Tensor::InFlow::HOST, api::INT32, {target_size});
        auto target_data = target.data<int32_t>();

        static const auto TO_REMOVE = 1;

        for (auto &box : bbox) {
            auto box_size = box.size(0);
            auto box_width = box.size(1);
            auto box_data = box.data<float>();
            for (int32_t n = 0; n < box_size; ++n) {
                auto area = (box_data[2] - box_data[0] + TO_REMOVE) * (box_data[3] - box_data[1] + TO_REMOVE);
                auto s = std::sqrt(area);
                auto target_lvls = std::floor(this->lvl0 + std::log2(s / this->s0 + this->eps));
                target_lvls = clamp<decltype(target_lvls)>(target_lvls, this->k_min, this->k_max);

                *target_data = int32_t(target_lvls) - this->k_min;

                box_data += box_width;
                target_data += 1;
            }
        }

        return target;
    }

};

/**
Arguments:
    x (list[Tensor]): feature maps for each level
    boxes (list[BoxList]): boxes to be used to perform the pooling operation.
Returns:
    result (Tensor)
Note:
    BoxList pack with repeating [image'size, boxes[-1, 4], scores]
 */
class ROIPooler : public api::Operator {
private:
    std::vector<int32_t> output_size;
    std::vector<float> scales;
    int sampling_ratio;

    std::vector<ROIAlign::shared> poolers;

    LevelMapper::shared map_levels;
public:
    template <typename T>
    static std::string to_string(const std::vector<T> &arr) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < arr.size(); ++i) {
            if (i) oss << ", ";
            oss << arr[i];
        }
        oss << "]";
        return oss.str();
    }

    void init(const api::OperatorParams &params, ts_OperatorContext *) override {
        output_size = api::tensor::array::to_int(params.get("output_size"));
        scales = api::tensor::array::to_float(params.get("scales"));
        sampling_ratio = api::tensor::to_int(params.get("sampling_ratio"));

        for (auto scale : scales) {
            poolers.emplace_back(std::make_shared<ROIAlign>(output_size, scale, sampling_ratio));
        }

        if (output_size.size() != 2 or output_size[0] != output_size[1]) {
            TS_THROW("ROIPooler: not supported output_size=" + to_string(output_size));
        }

        auto lvl_min = int(-std::log2(scales.front()));
        auto lvl_max = int(-std::log2(scales.back()));


        map_levels = std::make_shared<LevelMapper>(lvl_min, lvl_max);
    }

    std::vector<std::pair<api::DTYPE, api::Shape>> infer(const std::vector<api::Tensor> &args, ts_OperatorContext *) override {
        TS_THROW("Not implement");
        return std::vector<std::pair<api::DTYPE, api::Shape>>();
    }

    static api::Tensor convert_to_roi_format(const std::vector<api::Tensor> &boxes, int width = 3, int ind = 1) {
        auto size = int(boxes.size());
        std::vector<api::Tensor> bbox;
        for (auto i = ind; i < size; i += width) {
            bbox.emplace_back(boxes[i]);
        }
        auto concat_boxes = bbox.size() == 1 ? bbox[0] : api::intime::concat(bbox, 0);
        auto ids = api::Tensor(api::Tensor::InFlow::HOST, api::FLOAT32, {concat_boxes.size(0), 1});
        auto ids_data = ids.data<float>();
        auto iter_ids = 0;
        for (auto i = ind; i < size; i += width) {
            for (auto n = 0; n < boxes[i].size(0); ++n) {
                *ids_data++ = iter_ids;
            }
            ++iter_ids;
        }
        return api::intime::concat({ids, concat_boxes}, 1);
    }

    std::vector<api::Tensor> run(const std::vector<api::Tensor> &args, ts_OperatorContext *) override {
        if (args.size() != 2) TS_THROW("ROIPooler: input number must be 2");

        auto x = args[0].view(api::Tensor::InFlow::HOST).unpack();
        auto boxes = args[1].view(api::Tensor::InFlow::HOST).unpack();

        auto N = x[0].size(0);

        if (N != 1) {
            TS_THROW("ROIPooler: batch size must be 1");
        }

        auto num_levels = poolers.size();
        auto rois = convert_to_roi_format(boxes);

        if (num_levels == 1) {
            return {this->poolers[0]->forward(x[0], rois)};
        }

        if (num_levels > x.size()) {
            TS_THROW("ROIPooler: pooler count must smaller than (or equal to) features count.");
        }

        if ((map_levels->k_max - map_levels->k_min) >= num_levels) {
            TS_THROW("ROIPooler: pooling level out of controll.");
        }

        auto levels = map_levels->forward(boxes);

        auto num_rois = rois.size(0);
        // auto num_channels = x[0].size(1);
        // auto output_size = this->output_size[0];

        // auto result = api::Tensor(api::Tensor::InFlow::HOST, x[0].dtype(), {num_rois, num_channels, output_size, output_size});
        // std::memset(result.data(), 0, result.count() * api::type_bytes(result.dtype()));

        std::vector<std::vector<int32_t>> idx_in_level_summary(num_levels);

        auto levels_count = levels.size(0);
        auto levels_data = levels.data<int32_t>();
        for (int i = 0; i < levels_count; ++i) {
            auto level = *levels_data++;
            idx_in_level_summary[level].emplace_back(i);
        }

        api::Tensor idx_in_level_transpose(api::Tensor::InFlow::HOST, api::INT32, {num_rois});
        auto idx_data = idx_in_level_transpose.data<int32_t>();
        auto idx_shift = 0;
        for (auto &idx_in_level : idx_in_level_summary) {
            for (auto &idx : idx_in_level) {
                idx_data[idx] = idx_shift++;
            }
        }

        std::vector<api::Tensor> result(num_levels, nullptr);

        for (size_t level = 0; level < num_levels; ++level) {
            auto &per_level_feature = x[level];
            auto &pooler = *this->poolers[level];

            auto &idx_in_level = idx_in_level_summary[level];

            auto rois_per_level = api::intime::gather(rois, api::tensor::build(api::INT32, idx_in_level), 0);
            auto local_result = pooler.forward(per_level_feature, rois_per_level);

            result[level] = local_result;
        }

        auto final_result = api::intime::concat(result, 0);
        final_result = api::intime::gather(final_result, idx_in_level_transpose, 0);

        return {final_result};
    }
};

class BoxCoder {
public:
    using self = BoxCoder;
    using shared = std::shared_ptr<self>;
private:
    std::vector<float> weights;
    std::vector<float> inverse_weights;
    float bbox_xform_clip;
public:
    BoxCoder(const std::vector<float> &weights, float bbox_xform_clip = std::log(1000.0f / 16))
        : weights(weights), bbox_xform_clip(bbox_xform_clip) {
        this->inverse_weights = this->weights;
        for (auto &weight : this->inverse_weights) {
            weight = 1.0f / weight;
        }
    }

    api::Tensor decode(const api::Tensor &rel_codes, const api::Tensor &boxes) {
        /*"""
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """*/

        api::Tensor pred_boxes(api::Tensor::InFlow::HOST, api::FLOAT32, rel_codes.sizes());
        std::memset(pred_boxes.data(), 0, pred_boxes.count() * api::type_bytes(pred_boxes.dtype()));

        auto rel_codes_width = rel_codes.size(1);
        auto boxes_width = boxes.size(1);

        auto N = rel_codes.size(0);
        if (boxes.size(0) != N) {
            TS_THROW("BBox decoding failed.");
        }

        if (rel_codes_width % 4 != 0) {
            TS_THROW("BBox decoding failed with = " + std::to_string(rel_codes_width));
        }

        const auto TO_REMOVE = 1;

        for (int i = 0; i < N; ++i) {
            auto rel_codes_data = rel_codes.data<float>() + i * rel_codes_width;
            auto boxes_data = boxes.data<float>() + i * boxes_width;
            auto pred_boxes_data = pred_boxes.data<float>() + i * rel_codes_width;

            auto width = boxes_data[2] - boxes_data[0] + TO_REMOVE;
            auto height = boxes_data[3] - boxes_data[1] + TO_REMOVE;

            auto ctr_x = boxes_data[0] + 0.5f * width;
            auto ctr_y = boxes_data[1] + 0.5f * height;

            for (int j = 0; j < rel_codes_width - 3; j += 4) {
                auto dx = rel_codes_data[j + 0] * inverse_weights[0];
                auto dy = rel_codes_data[j + 1] * inverse_weights[1];
                auto dw = rel_codes_data[j + 2] * inverse_weights[2];
                auto dh = rel_codes_data[j + 3] * inverse_weights[3];

                if (dw > bbox_xform_clip) dw = bbox_xform_clip;
                if (dh > bbox_xform_clip) dh = bbox_xform_clip;

                auto pred_ctr_x = dx * width + ctr_x;
                auto pred_ctr_y = dy * height + ctr_y;
                auto pred_w = std::exp(dw) * width;
                auto pred_h = std::exp(dh) * height;

                // x1
                pred_boxes_data[j + 0] = pred_ctr_x - 0.5f * pred_w;
                // y1
                pred_boxes_data[j + 1] = pred_ctr_y - 0.5f * pred_h;
                // x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
                pred_boxes_data[j + 2] = pred_ctr_x + 0.5f * pred_w - 1;
                // y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
                pred_boxes_data[j + 3] = pred_ctr_y + 0.5f * pred_h - 1;
            }
        }

        return std::move(pred_boxes);
    }
};

/**
Arguments:
    x (tuple[tensor, tensor]): x contains the class logits
        and the box_regression from the model.
    boxes (list[BoxList]): bounding boxes that are used as
        reference, one for ech image

Returns:
    results (list[BoxList]): one BoxList for each image, containing
        the extra fields labels and scores
Note:
    boxes packed as repeating [image'size, boxes[-1, 4], objectness]
    results packed as repeating [image'size, boxes[-1, 4], scores, labels]
 */
class PostProcessor : public api::Operator {
private:
    std::vector<float> weights;
    float bbox_xform_clip = std::log(1000.0f / 16);

    float score_thresh = 0.7;
    float nms_thresh = 0.7;
    int32_t detections_per_img = -1;

    BoxCoder::shared box_coder;
public:
    void init(const api::OperatorParams &params, ts_OperatorContext *) override {
        auto weights_tensor = api::tensor::cast(api::FLOAT32, params.get("weights"));
        auto weights_data = weights_tensor.data<float>();
        auto weights_count = weights_tensor.count();
        this->weights = std::vector<float>(weights_data, weights_data + weights_count);
        this->bbox_xform_clip = api::tensor::to_float(params.get("bbox_xform_clip"));

        this->score_thresh = api::tensor::to_float(params.get("score_thresh"));
        this->nms_thresh = api::tensor::to_float(params.get("nms"));
        this->detections_per_img = api::tensor::to_int(params.get("detections_per_img"));

        box_coder = std::make_shared<BoxCoder>(this->weights, this->bbox_xform_clip);
    }

    std::vector<std::pair<api::DTYPE, api::Shape>> infer(const std::vector<api::Tensor> &args, ts_OperatorContext *) override {
        TS_THROW("Not implement");
        return std::vector<std::pair<api::DTYPE, api::Shape>>();
    }

    static std::vector<api::Tensor> prepare_boxlist(const api::Tensor &boxes, const api::Tensor &scores, const api::Tensor &image_shape) {
        return {image_shape, boxes.reshape({-1, 4}), scores.reshape({-1})};
    }

    template <typename T>
    static inline T clamp(T x, T min, T max) {
        return std::max(min, std::min(max, x));
    }

    static void clip_to_image(std::vector<api::Tensor> &boxlist, int width = 3, int shape_ind = 0, int boxes_ind = 1) {
        auto size = int(boxlist.size());

        static const auto WIDTH = 4;
        static const auto TO_REMOVE = 1;
        for (;shape_ind < size && boxes_ind < size; shape_ind += width, boxes_ind += width) {
            auto &shape = boxlist[shape_ind];
            auto &boxs = boxlist[boxes_ind];
            auto boxes_data = boxs.data<float>();

            auto int_shape = shape;
            if (int_shape.dtype() != api::INT32) {
                int_shape = api::tensor::cast(api::INT32, int_shape);
            }

            int w = int_shape.data<int32_t>(0);
            int h = int_shape.data<int32_t>(1);

            auto N = boxs.size(0);
            for (int i = 0; i < N; ++i) {
                auto box = boxes_data + i * WIDTH;
                // auto score = scores + i;
                box[0] = clamp<float>(box[0], 0, w - TO_REMOVE);
                box[1] = clamp<float>(box[1], 0, h - TO_REMOVE);
                box[2] = clamp<float>(box[2], 0, w - TO_REMOVE);
                box[3] = clamp<float>(box[3], 0, h - TO_REMOVE);
            }
        }
    }


    static float IoU(const float *w1, const float *w2) {
        auto xOverlap = std::max<float>(0, std::min(w1[2] - 1, w2[2] - 1) - std::max(w1[0], w2[0]) + 1);
        auto yOverlap = std::max<float>(0, std::min(w1[3] - 1, w2[3] - 1) - std::max(w1[1], w2[1]) + 1);
        auto intersection = xOverlap * yOverlap;

        auto w1_width = w1[2] - w1[0];
        auto w1_height = w1[3] - w1[1];
        auto w2_width = w2[2] - w2[0];
        auto w2_height = w2[3] - w2[1];

        auto unio = w1_width * w1_height + w2_width * w2_height - intersection;
        return float(intersection) / unio;
    }

    /**
     *
     * @param size
     * @param width
     * @param height
     * @param boxes
     * @param scores
     * @param min_size
     * @param nms_thresh
     * @param max_proposals
     * @return box, score
     */
    static std::vector<api::Tensor> nms(const api::Tensor &boxes, const api::Tensor &scores, float nms_thresh) {
        auto size = boxes.size(0);

        std::vector<bool> flag(size, false);    // remove flag
        static const auto WIDTH = 4;
        // static const auto TO_REMOVE = 1;

        auto boxes_data = boxes.data<float>();
        // auto scores_data = scores.data<float>();

        int count = 0;
        for (int i = 0; i < size; i++) {
            if (flag[i]) continue;
            else ++count;
            for (int j = i + 1; j < size; j++) {
                if (flag[j]) continue;
                if (IoU(&boxes_data[i * WIDTH], &boxes_data[j * WIDTH]) > nms_thresh) flag[j] = true;
            }
        }

        std::vector<int32_t> kept_ind;
        kept_ind.reserve(size / 2);

        int kept_size = 0;

        for (int i = 0; i < size; i++) {
            if (flag[i]) continue;
            kept_ind.push_back(i);

            ++kept_size;
        }

        auto kept_ind_tensor = api::tensor::build(api::INT32, kept_ind);
        auto kept_boxes = api::intime::gather(boxes, kept_ind_tensor, 0);
        auto kept_scores = api::intime::gather(scores, kept_ind_tensor, 0);

        return {kept_boxes, kept_scores};
    }

    static api::Tensor topk(const api::Tensor &source, int k) {
        auto dim0 = source.size(0);

        k = std::min<int>(k, dim0);

        api::Tensor index(api::Tensor::InFlow::HOST, api::INT32, {k});

        auto source_cpu = source.view(api::Tensor::InFlow::HOST);

        auto *index_data = index.data<int32_t>();
        int32_t index_size = 0;
        auto source_data = source_cpu.data<float>();

        auto index_compare = [&](int32_t lhs, int32_t rhs) -> bool {
            return source_data[lhs] > source_data[rhs];
        };

        for (int32_t n = 0; n < dim0; ++n) {
            if (index_size < k) {
                index_data[index_size++] = n;
                std::push_heap(index_data, index_data + index_size, index_compare);
            } else {
                std::pop_heap(index_data, index_data + index_size, index_compare);
                index_data[k - 1] = n;
                std::push_heap(index_data, index_data + index_size, index_compare);
            }
        }

        std::sort_heap(index_data, index_data + index_size, index_compare);

        return index;
    }


    std::vector<api::Tensor> filter_results(const std::vector<api::Tensor> &boxlist, int num_classes,
        int width = 3, int shape_ind = 0, int boxes_ind = 1, int score_ind = 2) {

        auto image_shape = boxlist[0];

        auto boxes = boxlist[boxes_ind].reshape({-1, num_classes, 4});
        auto scores = boxlist[score_ind].reshape({-1, num_classes});
        auto scores_data = scores.data<float>();

        auto N = scores.size(0);

        std::vector<std::vector<int32_t>> num_class_index_summary(num_classes);
        for (auto &list : num_class_index_summary) list.reserve(N / 4);

        for (int i = 0; i < N; ++i) {
            auto object_scores = &scores_data[i * num_classes];
            auto outer_shift = i * num_classes;
            for (int j = 1; j < num_classes; ++j) {
                auto score = object_scores[j];
                if (score <= this->score_thresh) continue;
                auto inner_shift = j;
                auto shift = outer_shift + inner_shift;
                num_class_index_summary[j].emplace_back(shift);
            }
        }


        auto flatten_boxes = boxes.reshape({-1, 4});
        auto flatten_scores = scores.reshape({-1});

        std::vector<api::Tensor> result;

        for (int j = 1; j < num_classes; ++j) {
            auto &index = num_class_index_summary[j];
            if (num_class_index_summary[j].empty()) continue;
            auto index_tensor = api::tensor::build(api::INT32, index);

            auto this_class_boxes = api::intime::gather(flatten_boxes, index_tensor, 0);
            auto this_class_scores = api::intime::gather(flatten_scores, index_tensor, 0);
            this_class_boxes.sync_cpu();
            this_class_scores.sync_cpu();

            auto boxlist_for_class = nms(this_class_boxes, this_class_scores, nms_thresh);
            auto num_labels = boxlist_for_class[1].size(0);

            api::Tensor field_label(api::Tensor::InFlow::HOST, api::INT32, {num_labels});
            auto label_data = field_label.data<int32_t>();
            for (int l = 0; l < num_labels; ++l) label_data[l] = j;

            result.push_back(image_shape);
            result.insert(result.end(), boxlist_for_class.begin(), boxlist_for_class.end());
            result.push_back(field_label);
        }

        if (result.empty()) {
            return {image_shape,
                    api::tensor::build(api::FLOAT32, {0, 4}),
                    api::tensor::build(api::FLOAT32, {0}),
                    api::tensor::build(api::INT32, {0})};
        }

        result = BoxSelector::cat_boxlist(result, 4, 1);
        auto number_of_detections = result[1].size(0);

        if (number_of_detections > detections_per_img && detections_per_img > 0) {
            auto index = topk(result[2], detections_per_img);
            for (size_t i = 1; i < result.size(); ++i) {
                auto &item = result[i];
                item = api::intime::gather(item, index, 0);
            }
        }

        return result;
    }

    std::vector<api::Tensor> run(const std::vector<api::Tensor> &args, ts_OperatorContext *) override {
        if (args.size() != 2) TS_THROW("PostProcessor: input number must be 2");

        auto x = args[0].unpack();
        auto boxes = args[1].unpack();

        if (boxes.size() != 3) {
            TS_THROW("PostProcessor: batch size must be 1");
        }

        auto &class_logits = x[0];
        auto &box_regression = x[1];

        auto class_prob = api::intime::softmax(class_logits, -1);

        /**
         * Following actions asumme that batch size is 1
         * Doing:
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
         */
        auto image_shapes = boxes[0];
        auto boxes_per_image = boxes[1].size(0);
        auto concat_boxes = boxes[1];

        auto proposals = box_coder->decode(
                box_regression.reshape({boxes_per_image, -1}).view(api::Tensor::InFlow::HOST),
                concat_boxes.view(api::Tensor::InFlow::HOST)
                );

        auto num_classes = class_prob.size(1);

        // Only for batch 1
        auto &prob = class_prob;
        auto &boxes_per_img = proposals;
        auto &image_shape = image_shapes;

        // using cpu memory
        prob.sync_cpu();
        boxes_per_img.sync_cpu();
        image_shape.sync_cpu();

        auto boxlist = prepare_boxlist(boxes_per_img, prob, image_shape);

        clip_to_image(boxlist);

        boxlist = filter_results(boxlist, num_classes);

        return boxlist;
    }
};

#endif //TENSORSTACK_PLUGIN_HPP
