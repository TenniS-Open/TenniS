//
// Created by kier on 2019/1/12.
//

#ifndef TENSORSTACK_BACKEDN_NAME_H
#define TENSORSTACK_BACKEDN_NAME_H

#include <string>

namespace ts {
    namespace name {
        using string = std::string;

        namespace layer {
            const string &dimshuffle();
            const string &transpose();
            const string &reshape();
            const string &conv2d();
            const string &padding_conv2d();
            const string &shape();
            const string &pad();
            const string &depthwise_conv2d();
            const string &padding_depthwise_conv2d();
            const string &add_bias();
            const string &batch_norm();
            const string &batch_scale();
            const string &fused_batch_norm();
            const string &add();
            const string &sub();
            const string &mul();
            const string &div();
            const string &inner_prod();
            const string &relu();
            const string &prelu();
            const string &relu_max();
            const string &sigmoid();
            const string &softmax();
            const string &concat();
            const string &flatten();
            const string &to_float();
            const string &pooling2d();
            const string &pooling2d_v2();
            const string &resize2d();
            const string &mx_pooling2d_padding();
        }

        extern string NCHW;
        extern string NHWC;
        extern string dim;
        extern string shuffle;
        extern string value;
        extern string permute;
        extern string shape;
        extern string format;
        extern string padding;
        extern string padding_value;
        extern string stride;
        extern string dialations;
        extern string epsilon;
        extern string max;
        extern string slope;
        extern string type;
        extern string padding_type;
        extern string ksize;


        extern string valid;
    }

}


#endif //TENSORSTACK_BACKEDN_NAME_H
