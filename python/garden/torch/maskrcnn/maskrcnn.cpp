//
// Created by kier on 19-5-13.
//


#include <string>
#include <cmath>
#include <algorithm>
#include <cstring>

#include "api/cpp/operator.h"
#include "api/cpp/intime.h"

using namespace ts;


class AnchorGenerator : public api::Operator {
private:
    std::vector<int> strides;
    std::vector<api::Tensor> cell_anchors;

public:
    void init(const api::OperatorParams &params) override {
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

    std::vector<std::pair<api::DTYPE, api::Shape>> infer(const std::vector<api::Tensor> &args) override {
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

    std::vector<api::Tensor> run(const std::vector<api::Tensor> &args) override {
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
    void init(const api::OperatorParams &params) override {
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

    std::vector<std::pair<api::DTYPE, api::Shape>> infer(const std::vector<api::Tensor> &args) override {
        TS_THROW("Not implement");
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
    std::vector<api::Tensor> nms(int size,
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

    std::vector<api::Tensor> run(const std::vector<api::Tensor> &args) override {
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