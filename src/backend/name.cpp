//
// Created by kier on 2019/1/12.
//

#include "backend/name.h"

namespace ts {
    namespace name {
        namespace layer {
            const string &dimshuffle() { static string str = "_dimshuffle"; return str; }
            const string &transpose() { static string str = "_transpose"; return str; }
            const string &reshape() { static string str = "_reshape"; return str; }
            const string &conv2d() { static string str = "conv2d"; return str; }
            const string &padding_conv2d() { static string str = "padding_conv2d"; return str; }
            const string &shape() { static string str = "_shape"; return str; }
            const string &pad() { static string str = "pad"; return str; }
            const string &depthwise_conv2d() { static string str = "depthwise_conv2d"; return str; }
            const string &padding_depthwise_conv2d() { static string str = "padding_depthwise_conv2d"; return str; }
            const string &add_bias() { static string str = "add_bias"; return str; }
            const string &batch_norm() { static string str = "batch_norm"; return str; }
            const string &batch_scale() { static string str = "batch_scale"; return str; }
            const string &fused_batch_norm() { static string str = "fused_batch_norm"; return str; }
            const string &add() { static string str = "add"; return str; }
            const string &sub() { static string str = "sub"; return str; }
            const string &mul() { static string str = "mul"; return str; }
            const string &div() { static string str = "div"; return str; }
            const string &inner_prod() { static string str = "inner_prod"; return str; }
            const string &relu() { static string str = "relu"; return str; }
            const string &prelu() { static string str = "prelu"; return str; }
            const string &relu_max() { static string str = "relu_max"; return str; }
            const string &sigmoid() { static string str = "sigmoid"; return str; }
            const string &softmax() { static string str = "softmax"; return str; }
            const string &concat() { static string str = "concat"; return str; }
            const string &flatten() { static string str = "flatten"; return str; }
            const string &to_float() { static string str = "to_float"; return str; }
            const string &pooling2d() { static string str = "pooling2d"; return str; }
            const string &pooling2d_v2() { static string str = "pooling2d_v2"; return str; }
            const string &resize2d() { static string str = "resize2d"; return str; }
            const string &mx_pooling2d_padding() { static string str = "_mx_pooling2d_padding"; return str; }
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
        string dialations = "dialations";
        string epsilon = "epsilon";
        string max = "max";
        string slope = "slope";
        string type = "type";
        string padding_type = "padding_type";
        string ksize = "ksize";
        string valid = "valid";

    }
}