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
            extern string dimshuffle;
            extern string transpose;
            extern string reshape;
            extern string conv2d;
            extern string padding_conv2d;
            extern string shape;
            extern string pad;
            extern string depthwise_conv2d;
            extern string padding_depthwise_conv2d;
            extern string add_bias;
            extern string batch_norm;
            extern string batch_scale;
            extern string fused_batch_norm;
            extern string add;
            extern string sub;
            extern string mul;
            extern string div;
            extern string inner_prod;
            extern string relu;
            extern string prelu;
            extern string relu_max;
            extern string sigmoid;
            extern string softmax;
            extern string concat;
            extern string flatten;
            extern string to_float;
            extern string pooling2d;
            extern string pooling2d_v2;
            extern string resize2d;
            extern string mx_pooling2d_padding;
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
