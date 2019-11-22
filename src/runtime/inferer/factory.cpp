//
// Created by kier on 2019/11/22.
//

#include <core/tensor_builder.h>
#include "global/shape_inferer_factory.h"
#include "utils/static.h"
#include "backend/name.h"

#include "runtime/inferer.h"

namespace ts {
    namespace infer_factory {
        static Tensor get_value(const Node &node) {
            if (node->op() == Bubble::Const) {
                return node->get("value");
            }
            if (node->has("#value")) {
                return node->get("#value");
            }
            return Tensor();
        }

        static void infer_const_value(const Node &node, const std::vector<bool> &ignore) {
            infer_value(*const_cast<Node *>(&node), ignore);
        }

        static TensorPrototype _param(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (!node->has("#shape")) {
                throw Exception(node->op() + ":" + node->name() + " must set #shape");
                // return TensorPrototype();
            }
            auto dtype = FLOAT32;
            if (node->has("#dtype")) {
                dtype = DTYPE(tensor::to_int(node->get("#dtype")));
            }
            auto shape = tensor::array::to_int(node->get("#shape"));
            return TensorPrototype(dtype, shape);
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "<param>", _param)

        static TensorPrototype _const(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto value = node->get("value");
            return value.proto();
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "<const>", _const)

        static TensorPrototype _resize2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto x = inputs[0];
            auto size_value = get_value(node.input(1));
            if (size_value.empty()) return VOID;
            auto size = tensor::array::to_int(size_value);
            auto y_shape = x.sizes();
            if (size.size() != y_shape.size()) return VOID;
            for (size_t i = 0; i < size.size(); ++i) {
                if (size[i] > 0) y_shape[i] = size[i];
            }
            return TensorPrototype(x.dtype(), y_shape);
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_resize2d", _resize2d)

        static TensorPrototype _transpose(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto x = inputs[0];
            auto permute = node->get_int_list("permute");
            auto y_shape = std::vector<int32_t>(permute.size());

            for (size_t i = 0; i < permute.size(); ++i) {
                if (permute[i] >= x.dims()) return VOID;
                y_shape[i] = x.size(permute[i]);
            }

            return TensorPrototype(x.dtype(), y_shape);
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_transpose", _transpose)

        static TensorPrototype _copy(const Node &node, const std::vector<TensorPrototype> &inputs) {
            return inputs[0];
        }

        static TensorPrototype to_float(const Node &node, const std::vector<TensorPrototype> &inputs) {
            return {FLOAT32, inputs[0].sizes()};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "to_float", to_float)

        static TensorPrototype crop_nd(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto x = inputs[0];
            auto size_value = get_value(node.input(1));
            if (size_value.empty()) return VOID;
            auto size = tensor::array::to_int(size_value);
            auto y_shape = x.sizes();
            if (size.size() != y_shape.size()) return VOID;
            for (size_t i = 0; i < size.size(); ++i) {
                if (size[i] > 0) y_shape[i] = size[i];
            }
            return TensorPrototype(x.dtype(), y_shape);
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "crop_nd", crop_nd)

        static int32_t conv2d_forward(int32_t x, int32_t padding, int32_t dilation, int32_t kernel, int32_t stride) {
            return int32_t(std::floor((x + padding - (dilation * (kernel - 1) + 1)) / stride + 1));
        }

        static TensorPrototype conv2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            auto stide = node->get_int_list("stride");
            auto dilation = node->get_int_list("dilation");
            auto padding = node->get_int_list("padding");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];
            auto &w = inputs[1];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = w.size(0);

            int32_t kernel_shape[] = {w.size(2), w.size(3)};

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = conv2d_forward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        dilation[dim],
                        kernel_shape[i],
                        stide[dim]);
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "conv2d", conv2d)

        TS_STATIC_ACTION(ShapeInferer::Register, "add_bias", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "relu", _copy)

        static int32_t pooling2d_forward(int32_t x, int32_t padding, int32_t kernel, int32_t stride) {
            return int32_t(std::ceil((x + padding - kernel) / float(stride) + 1));
        }

        static TensorPrototype pooling2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            auto stide = node->get_int_list("stride");
            auto padding = node->get_int_list("padding");
            auto ksize = node->get_int_list("ksize");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = x.size(channel_dims);

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = pooling2d_forward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        ksize[dim],
                        stide[dim]);
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "pooling2d", pooling2d)

        static int32_t eltwise_infer_dim(int32_t a, int32_t b) {

            if (a <= 0) {
                if (b == 1)
                    return -1;
                else
                    return b;
            }
            if (a == 1)
                return b;
            if (b <= 0)
                return a;
            if (b == 1)
                return a;
            if (a == b)
                return a;
            return -1;
        }

        static void begin_insert_ones(std::vector<int32_t> &x, size_t n) {
            auto ones = std::vector<int32_t>(n, 1);
            x.insert(x.begin(), ones.begin(), ones.end());
        }

        static TensorPrototype _eltwise(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto dtype = inputs[0].dtype();

            auto lhs_shape = inputs[0].sizes();
            auto rhs_shape = inputs[1].sizes();

            if (lhs_shape.size() > rhs_shape.size()) {
                begin_insert_ones(rhs_shape, lhs_shape.size() - rhs_shape.size());
            } else if (rhs_shape.size() > lhs_shape.size()) {
                begin_insert_ones(lhs_shape, rhs_shape.size() - lhs_shape.size());
            }

            auto dims = lhs_shape.size();
            auto out_shape = std::vector<int32_t>(dims, -1);

            for (size_t i = 0; i < dims; ++i) {
                out_shape[i] = eltwise_infer_dim(lhs_shape[i], rhs_shape[i]);
            }

            return {dtype, out_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "add", _eltwise)
        TS_STATIC_ACTION(ShapeInferer::Register, "sub", _eltwise)
        TS_STATIC_ACTION(ShapeInferer::Register, "mul", _eltwise)
        TS_STATIC_ACTION(ShapeInferer::Register, "div", _eltwise)

        static TensorPrototype flatten(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            if (x.dims() < 1) return x;
            if (x.dims() < 2) { return {x.dtype(), {x.size(0), 1}}; }

            int32_t prod = 1;
            for (auto it = x.sizes().begin() + 1; it != x.sizes().end(); ++it) {
                if (*it < 0) {
                    prod = -1;
                    break;
                }
                prod *= *it;
            }

            return {x.dtype(), {x.size(0), prod}};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "flatten", flatten)

        static TensorPrototype inner_prod(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];
            auto &w = inputs[1];

            bool transpose = false;
            if (node->has("transpose")) {
                transpose = node->get_bool("transpose");
            }

            if (transpose) {
                return {x.dtype(), {x.size(0), w.size(0)}};
            } else {
                return {x.dtype(), {x.size(0), w.size(1)}};
            }
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "inner_prod", inner_prod)

        static bool valid_dims(const std::vector<int32_t> &dims) {
            for (auto dim : dims) {
                if (dim <= 0) return false;
            }
            return true;
        }

        static TensorPrototype _reshape(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];
            auto shape = node->get_int_list("shape");

            for (size_t i = 0; i < shape.size(); ++i) {
                if (shape[i] == 0) {
                    if (i >= x.dims()) {
                        return VOID;
                    }
                    shape[i] = x.size(i);
                }
            }

            if (valid_dims(x.sizes())) {
                Tensor tmp(INT8, x.sizes());
                tmp = tmp.reshape(shape);
                shape = tmp.sizes();
            }

            return {x.dtype(), shape};

        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_reshape", _reshape)

        TS_STATIC_ACTION(ShapeInferer::Register, "softmax", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "batch_norm", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "batch_scale", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "fused_batch_norm", _copy)

        static TensorPrototype _cast(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto dtype = node->get_int("dtype");
            return {DTYPE(dtype), inputs[0].sizes()};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_cast", _cast)

        static TensorPrototype dynamic_padding(const Node &node, const std::vector<TensorPrototype> &inputs) {
            infer_const_value(node, {true});
            return {INT32, {4, 2}};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_onnx_pooling2d_padding", dynamic_padding)
        TS_STATIC_ACTION(ShapeInferer::Register, "_dragon_pooling2d_padding", dynamic_padding)
        TS_STATIC_ACTION(ShapeInferer::Register, "_mx_pooling2d_padding", dynamic_padding)
        TS_STATIC_ACTION(ShapeInferer::Register, "_tf_conv2d_padding", dynamic_padding)
        TS_STATIC_ACTION(ShapeInferer::Register, "_tf_pooling2d_padding", dynamic_padding)
        TS_STATIC_ACTION(ShapeInferer::Register, "_dragon_conv2d_padding", dynamic_padding)

        static TensorPrototype pooling2d_v2(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            auto padding_value = get_value(node.input(1));
            if (padding_value.empty()) return VOID;
            auto stride_value = get_value(node.input(2));
            if (stride_value.empty()) return VOID;
            auto ksize_value = get_value(node.input(3));
            if (ksize_value.empty()) return VOID;

            auto padding = tensor::array::to_int(padding_value);
            auto stide = tensor::array::to_int(stride_value);
            auto ksize = tensor::array::to_int(ksize_value);

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = x.size(channel_dims);

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = pooling2d_forward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        ksize[dim],
                        stide[dim]);
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "pooling2d_v2", pooling2d_v2)

        static TensorPrototype conv2d_v2(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            auto padding_value = get_value(node.input(1));
            auto padding = tensor::array::to_int(padding_value);

            auto stide = node->get_int_list("stride");
            auto dilation = node->get_int_list("dilation");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];
            auto &w = inputs[2];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = w.size(0);

            int32_t kernel_shape[] = {w.size(2), w.size(3)};

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = conv2d_forward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        dilation[dim],
                        kernel_shape[i],
                        stide[dim]);
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "conv2d_v2", conv2d_v2)

        static TensorPrototype gemm(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto A = inputs[0];
            auto B = inputs[1];


            auto transA = node->get_bool("transA");
            auto transB = node->get_bool("transB");

            auto M = transA ? A.size(1) : A.size(0);
            auto N = transB ? B.size(0) : B.size(1);

            return {A.dtype(), {M, N}};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "gemm", gemm)

        static TensorPrototype concat(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.empty()) return VOID;

            auto dim = node->get_int("dim");

            auto dtype = inputs[0].dtype();
            auto shape = inputs[0].sizes();

            if (dim < 0) dim = int32_t(shape.size()) + dim;
            if (dim < 0 || dim >= int32_t(shape.size())) return VOID;

            for (size_t i = 1; i < inputs.size(); ++i) {
                auto inc = inputs[i].size(dim);
                if (inc < 0) {
                    shape[dim] = -1;
                    break;
                }
                shape[dim] += inc;
            }

            return {dtype, shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "concat", concat)

        static TensorPrototype global_pooling2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = x.size(channel_dims);

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                y_shape[dim] = 1;
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "global_pooling2d", global_pooling2d)

        TS_STATIC_ACTION(ShapeInferer::Register, "sigmoid", _copy)

        static TensorPrototype _dims(const Node &node, const std::vector<TensorPrototype> &inputs) {
            infer_const_value(node, {true});
            return {INT32, {}};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_dims", _dims)

        static TensorPrototype _expand(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto x = inputs[0];

            auto dims_value = get_value(node.input(1));
            if (dims_value.empty()) return VOID;

            auto dims = size_t(tensor::to_int(dims_value));

            auto front = int32_t(dims);
            auto end = int32_t(dims);
            bool inverse = false;
            if (node->has("front")) front = node->get_int("front");
            if (node->has("end")) end = node->get_int("end");
            if (node->has("inverse")) inverse = node->get_bool("inverse");

            auto y = x.sizes();

            if (!inverse) {
                while (y.size() < dims && front > 0) {
                    y.insert(y.begin(), 1);
                }
                while (y.size() < dims && end > 0) {
                    y.insert(y.end(), 1);
                }
            } else {
                while (y.size() < dims && end > 0) {
                    y.insert(y.end(), 1);
                }
                while (y.size() < dims && front > 0) {
                    y.insert(y.begin(), 1);
                }
            }

            return {x.dtype(), y};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_expand", _expand)
    }

}
