//
// Created by kier on 2019/1/12.
//

#include "backend/name.h"

namespace ts {
    namespace name {
        namespace layer {
            string dimshuffle = "_dimshuffle";
            string transpose = "_transpose";
            string reshape = "_reshape";
            string conv2d = "conv2d";
            string padding_conv2d = "padding_conv2d";
            string shape = "_shape";
            string pad = "pad";
            string depthwise_conv2d = "depthwise_conv2d";
            string padding_depthwise_conv2d = "padding_depthwise_conv2d";
            string add_bias = "add_bias";
            string batch_norm = "batch_norm";
            string batch_scale = "batch_scale";
            string fused_batch_norm = "fused_batch_norm";
            string add = "add";
            string sub = "sub";
            string mul = "mul";
            string div = "div";
            string inner_prod = "inner_prod";
            string relu = "relu";
            string prelu = "prelu";
            string relu_max = "relu_max";
            string sigmoid = "sigmoid";
            string softmax = "softmax";
            string concat = "concat";
            string flatten = "flatten";
            string to_float = "to_float";
            string pooling2d = "pooling2d";
            string pooling2d_v2 = "pooling2d_v2";
            string resize2d = "resize2d";
            string mx_pooling2d_padding = "_mx_pooling2d_padding";
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