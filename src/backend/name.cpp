//
// Created by kier on 2019/1/12.
//

#include <backend/name.h>

#include "backend/name.h"

namespace ts {
    namespace name {
        namespace layer {
            const string &field() TS_NOEXCEPT { static string str = "_field"; return str; }
            const string &pack() TS_NOEXCEPT { static string str = "_pack"; return str; }
            const string &dimshuffle() TS_NOEXCEPT { static string str = "_dimshuffle"; return str; }
            const string &transpose() TS_NOEXCEPT { static string str = "_transpose"; return str; }
            const string &reshape() TS_NOEXCEPT { static string str = "_reshape"; return str; }
            const string &conv2d() TS_NOEXCEPT { static string str = "conv2d"; return str; }
            const string &conv2d_v2() TS_NOEXCEPT { static string str = "conv2d_v2"; return str; }
            const string &shape() TS_NOEXCEPT { static string str = "_shape"; return str; }
            const string &pad() TS_NOEXCEPT { static string str = "pad"; return str; }
            const string &depthwise_conv2d() TS_NOEXCEPT { static string str = "depthwise_conv2d"; return str; }
            const string &depthwise_conv2d_v2() TS_NOEXCEPT { static string str = "depthwise_conv2d_v2"; return str; }
            const string &add_bias() TS_NOEXCEPT { static string str = "add_bias"; return str; }
            const string &batch_norm() TS_NOEXCEPT { static string str = "batch_norm"; return str; }
            const string &batch_scale() TS_NOEXCEPT { static string str = "batch_scale"; return str; }
            const string &fused_batch_norm() TS_NOEXCEPT { static string str = "fused_batch_norm"; return str; }
            const string &add() TS_NOEXCEPT { static string str = "add"; return str; }
            const string &sub() TS_NOEXCEPT { static string str = "sub"; return str; }
            const string &mul() TS_NOEXCEPT { static string str = "mul"; return str; }
            const string &div() TS_NOEXCEPT { static string str = "div"; return str; }
            const string &inner_prod() TS_NOEXCEPT { static string str = "inner_prod"; return str; }
            const string &relu() TS_NOEXCEPT { static string str = "relu"; return str; }
            const string &prelu() TS_NOEXCEPT { static string str = "prelu"; return str; }
            const string &relu_max() TS_NOEXCEPT { static string str = "relu_max"; return str; }
            const string &sigmoid() TS_NOEXCEPT { static string str = "sigmoid"; return str; }
            const string &softmax() TS_NOEXCEPT { static string str = "softmax"; return str; }
            const string &concat() TS_NOEXCEPT { static string str = "concat"; return str; }
            const string &flatten() TS_NOEXCEPT { static string str = "flatten"; return str; }
            const string &to_float() TS_NOEXCEPT { static string str = "to_float"; return str; }
            const string &pooling2d() TS_NOEXCEPT { static string str = "pooling2d"; return str; }
            const string &pooling2d_v2() TS_NOEXCEPT { static string str = "pooling2d_v2"; return str; }
            const string &resize2d() TS_NOEXCEPT { static string str = "_resize2d"; return str; }
            const string &mx_pooling2d_padding() TS_NOEXCEPT { static string str = "_mx_pooling2d_padding"; return str; }
            const string &nhwc_center_crop2d() TS_NOEXCEPT { static string str = "_nhwc_center_crop2d"; return str; }
            const string &cast() TS_NOEXCEPT { static string str = "_cast"; return str; }
            const string &onnx_pooling2d_padding() TS_NOEXCEPT { static string str = "_onnx_pooling2d_padding"; return str; }
            const string &gather() TS_NOEXCEPT { static string str = "gather"; return str; }
            const string &unsqueeze() TS_NOEXCEPT { static string str = "unsqueeze"; return str; }
            const string &gemm() TS_NOEXCEPT { static string str = "gemm"; return str; }
            const string &reshape_v2() TS_NOEXCEPT { static string str = "_reshape_v2"; return str; }

            const string &copy() TS_NOEXCEPT {
                static string str = "_copy";
                return str;
            }
        }

        namespace typo {
            string dialations = "dialations";
        }

        string NCHW = "NCHW";
        string NHWC = "NHWC";
        string dim = "dim";
        string shuffle = "shuffle";
        string value = "value";
        string permute = "permute";
        string shape = "shape";
        string format = "format";
        string padding = "padding";
        string padding_value = "padding_value";
        string stride = "stride";
        string dilation = "dilation";
        string epsilon = "epsilon";
        string max = "max";
        string slope = "slope";
        string type = "type";
        string padding_type = "padding_type";
        string ksize = "ksize";
        string valid = "valid";
        string device = "device";
        string offset = "offset";
		string smooth = "smooth";
        string size = "size";
        string prewhiten = "prewhiten";
        string dtype = "dtype";
        string output_shape = "output_shape";

        string auto_pad = "auto_pad";
        string axis = "axis";
        string axes = "axes";
        string NOTSET = "NOTSET";
        string SAME_UPPER = "SAME_UPPER";
        string SAME_LOWER = "SAME_LOWER";
        string VALID = "VALID";
        string alpha = "alpha";
        string beta = "beta";
        string transA = "transA";
        string transB = "transB";
    }
}