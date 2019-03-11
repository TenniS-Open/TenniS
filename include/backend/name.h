//
// Created by kier on 2019/1/12.
//

#ifndef TENSORSTACK_BACKEDN_NAME_H
#define TENSORSTACK_BACKEDN_NAME_H

#include <string>
#include "utils/except.h"

namespace ts {
    namespace name {
        using string = std::string;

        namespace layer {
            const string &field() TS_NOEXCEPT;
            const string &pack() TS_NOEXCEPT;
            const string &dimshuffle() TS_NOEXCEPT;
            const string &transpose() TS_NOEXCEPT;
            const string &reshape() TS_NOEXCEPT;
            const string &conv2d() TS_NOEXCEPT;
            const string &conv2d_v2() TS_NOEXCEPT;
            const string &shape() TS_NOEXCEPT;
            const string &pad() TS_NOEXCEPT;
            const string &depthwise_conv2d() TS_NOEXCEPT;
            const string &depthwise_conv2d_v2() TS_NOEXCEPT;
            const string &add_bias() TS_NOEXCEPT;
            const string &batch_norm() TS_NOEXCEPT;
            const string &batch_scale() TS_NOEXCEPT;
            const string &fused_batch_norm() TS_NOEXCEPT;
            const string &add() TS_NOEXCEPT;
            const string &sub() TS_NOEXCEPT;
            const string &mul() TS_NOEXCEPT;
            const string &div() TS_NOEXCEPT;
            const string &inner_prod() TS_NOEXCEPT;
            const string &relu() TS_NOEXCEPT;
            const string &prelu() TS_NOEXCEPT;
            const string &relu_max() TS_NOEXCEPT;
            const string &sigmoid() TS_NOEXCEPT;
            const string &softmax() TS_NOEXCEPT;
            const string &concat() TS_NOEXCEPT;
            const string &flatten() TS_NOEXCEPT;
            const string &to_float() TS_NOEXCEPT;
            const string &pooling2d() TS_NOEXCEPT;
            const string &pooling2d_v2() TS_NOEXCEPT;
            const string &resize2d() TS_NOEXCEPT;
            const string &mx_pooling2d_padding() TS_NOEXCEPT;
            const string &copy() TS_NOEXCEPT;
            const string &nhwc_center_crop2d() TS_NOEXCEPT;
            const string &cast() TS_NOEXCEPT;

            const string &onnx_pooling2d_padding() TS_NOEXCEPT;
            const string &gather() TS_NOEXCEPT;
            const string &unsqueeze() TS_NOEXCEPT;
            const string &gemm() TS_NOEXCEPT;

            const string &reshape_v2() TS_NOEXCEPT;

            // 2019-03-11
            const string &global_pooling2d() TS_NOEXCEPT;
        }

        namespace typo {
            extern string dialations;
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
        extern string dilation;
        extern string epsilon;
        extern string max;
        extern string slope;
        extern string type;
        extern string padding_type;
        extern string ksize;
        extern string device;
        extern string offset;
		extern string smooth;
        extern string size;
        extern string prewhiten;
        extern string dtype;
        extern string output_shape;

        extern string valid;

        extern string auto_pad;
        extern string axis;
        extern string axes;
        extern string NOTSET;
        extern string SAME_UPPER;
        extern string SAME_LOWER;
        extern string VALID;
        extern string alpha;
        extern string beta;
        extern string transA;
        extern string transB;
    }

}


#endif //TENSORSTACK_BACKEDN_NAME_H
