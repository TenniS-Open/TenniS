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
:return: Repeating [[image_width, image_height], Anchor[-1, 4]]
 */
class BoxSelector : public api::Operator {
private:
    int pre_nms_top_n = 100;
    std::vector<float> weights;
    float bbox_xform_clip = std::log(1000.0f / 16);
public:
    void init(const api::OperatorParams &params) override {
        this->pre_nms_top_n = api::tensor::to_int(params.get("pre_nms_top_n"));
        auto weights_tensor = api::tensor::cast(api::FLOAT32, params.get("weights"));
        auto weights_data = weights_tensor.data<float>();
        auto weights_count = weights_tensor.count();
        this->weights = std::vector<float>(weights_data, weights_data + weights_count);
        this->bbox_xform_clip = api::tensor::to_float(params.get("bbox_xform_clip"));
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
    static std::tuple<api::Tensor, api::Tensor> torch_topk_dim1(const api::Tensor &source, int k) {
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

    static api::Tensor torch_batch_slice(const api::Tensor &tensor3d_xx4, const api::Tensor &idx) {
        return tensor3d_xx4;
    }

    /**
     *
     * @param anchors list[BoxList]
     * @param objectness tensor of size N, A, H, W
     * @param box_regression tensor of size N, A * 4, H, W
     */
    void forward_for_single_feature_map(
            const std::vector<api::Tensor> &anchors,
            api::Tensor objectness, api::Tensor box_regression) {
        auto NAHW = ts_Tensor_shape(objectness.get_raw());
        auto N = NAHW[0];
        auto A = NAHW[1];
        auto H = NAHW[2];
        auto W = NAHW[3];

        objectness = api::intime::transpose(objectness, {0, 2, 3, 1}).reshape({N, -1});
        objectness = api::intime::sigmoid(objectness);

        box_regression = api::intime::transpose(box_regression.reshape({N, -1, 4, H, W}), {0, 3, 4, 1, 2});
        box_regression = box_regression.reshape({N, -1, 4});

        auto num_anchors = A * H * W;
        auto pre_nms_top_n = std::min<int32_t>(this->pre_nms_top_n, num_anchors);

        api::Tensor topk_idx = nullptr;

        auto _0 = torch_topk_dim1(objectness, pre_nms_top_n);
        objectness = std::get<0>(_0);
        topk_idx = std::get<1>(_0);
        topk_idx = topk_idx.reshape({topk_idx.count()});

        box_regression = api::intime::gather(box_regression, topk_idx, 1);

        std::vector<api::Tensor> list_anchors;
        auto list_anchors_size = anchors.size() / 2;
        for (size_t i = 0; i < list_anchors_size; ++i) {
            list_anchors.push_back(anchors[i * 2 + 1]);
        }

        auto concat_anchors = api::intime::concat(list_anchors, 0).reshape({N, -1, 4});
        concat_anchors = api::intime::gather(concat_anchors, topk_idx, 1);

        api::Tensor proposals = nullptr;
    }

    std::vector<api::Tensor> run(const std::vector<api::Tensor> &args) override {
        if (args.size() != 3) TS_THROW("AnchorGenerator: input number must be 3");
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

        for (size_t i = 0; i <  num_levels; ++i) {
            auto &a = anchors[i];
            auto &o = objectness[i];
            auto &b = box_regression[i];
            forward_for_single_feature_map(a, o, b);
        }
    }
};