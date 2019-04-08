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
            TS_DEBUG_API const string &field() TS_NOEXCEPT;
            TS_DEBUG_API const string &pack() TS_NOEXCEPT;
            TS_DEBUG_API const string &dimshuffle() TS_NOEXCEPT;
            TS_DEBUG_API const string &transpose() TS_NOEXCEPT;
            TS_DEBUG_API const string &reshape() TS_NOEXCEPT;
            TS_DEBUG_API const string &conv2d() TS_NOEXCEPT;
            TS_DEBUG_API const string &conv2d_v2() TS_NOEXCEPT;
            TS_DEBUG_API const string &shape() TS_NOEXCEPT;
            TS_DEBUG_API const string &pad() TS_NOEXCEPT;
            TS_DEBUG_API const string &depthwise_conv2d() TS_NOEXCEPT;
            TS_DEBUG_API const string &depthwise_conv2d_v2() TS_NOEXCEPT;
            TS_DEBUG_API const string &add_bias() TS_NOEXCEPT;
            TS_DEBUG_API const string &batch_norm() TS_NOEXCEPT;
            TS_DEBUG_API const string &batch_scale() TS_NOEXCEPT;
            TS_DEBUG_API const string &fused_batch_norm() TS_NOEXCEPT;
            TS_DEBUG_API const string &add() TS_NOEXCEPT;
            TS_DEBUG_API const string &sub() TS_NOEXCEPT;
            TS_DEBUG_API const string &mul() TS_NOEXCEPT;
            TS_DEBUG_API const string &div() TS_NOEXCEPT;
            TS_DEBUG_API const string &inner_prod() TS_NOEXCEPT;
            TS_DEBUG_API const string &relu() TS_NOEXCEPT;
            TS_DEBUG_API const string &prelu() TS_NOEXCEPT;
            TS_DEBUG_API const string &relu_max() TS_NOEXCEPT;
            TS_DEBUG_API const string &sigmoid() TS_NOEXCEPT;
            TS_DEBUG_API const string &softmax() TS_NOEXCEPT;
            TS_DEBUG_API const string &concat() TS_NOEXCEPT;
            TS_DEBUG_API const string &flatten() TS_NOEXCEPT;
            TS_DEBUG_API const string &to_float() TS_NOEXCEPT;
            TS_DEBUG_API const string &pooling2d() TS_NOEXCEPT;
            TS_DEBUG_API const string &pooling2d_v2() TS_NOEXCEPT;
            TS_DEBUG_API const string &resize2d() TS_NOEXCEPT;
            TS_DEBUG_API const string &mx_pooling2d_padding() TS_NOEXCEPT;
            TS_DEBUG_API const string &copy() TS_NOEXCEPT;
            TS_DEBUG_API const string &nhwc_center_crop2d() TS_NOEXCEPT;
            TS_DEBUG_API const string &cast() TS_NOEXCEPT;

            TS_DEBUG_API const string &onnx_pooling2d_padding() TS_NOEXCEPT;
            TS_DEBUG_API const string &gather() TS_NOEXCEPT;
            TS_DEBUG_API const string &unsqueeze() TS_NOEXCEPT;
            TS_DEBUG_API const string &gemm() TS_NOEXCEPT;

            TS_DEBUG_API const string &reshape_v2() TS_NOEXCEPT;

            // 2019-03-11
            TS_DEBUG_API const string &global_pooling2d() TS_NOEXCEPT;
            // 2019-03-12
            TS_DEBUG_API const string &limit() TS_NOEXCEPT;
            // 2019-03-14
            TS_DEBUG_API const string &shape_index_patch() TS_NOEXCEPT;
            // 2019-03-17
            TS_DEBUG_API const string &tf_pooling2d_padding() TS_NOEXCEPT;
            TS_DEBUG_API const string &tf_conv2d_padding() TS_NOEXCEPT;
            // 2019-03-18
            TS_DEBUG_API const string &nhwc_scale_resize2d() TS_NOEXCEPT;
        }

        namespace typo {
            TS_DEBUG_API extern string dialations;
        }

        TS_DEBUG_API extern string NCHW;
        TS_DEBUG_API extern string NHWC;
        TS_DEBUG_API extern string dim;
        TS_DEBUG_API extern string shuffle;
        TS_DEBUG_API extern string value;
        TS_DEBUG_API extern string permute;
        TS_DEBUG_API extern string shape;
        TS_DEBUG_API extern string format;
        TS_DEBUG_API extern string padding;
        TS_DEBUG_API extern string padding_value;
        TS_DEBUG_API extern string stride;
        TS_DEBUG_API extern string dilation;
        TS_DEBUG_API extern string epsilon;
        TS_DEBUG_API extern string max;
        TS_DEBUG_API extern string slope;
        TS_DEBUG_API extern string type;
        TS_DEBUG_API extern string padding_type;
        TS_DEBUG_API extern string ksize;
        TS_DEBUG_API extern string device;
        TS_DEBUG_API extern string offset;
        TS_DEBUG_API extern string smooth;
        TS_DEBUG_API extern string size;
        TS_DEBUG_API extern string prewhiten;
        TS_DEBUG_API extern string dtype;
        TS_DEBUG_API extern string output_shape;

        TS_DEBUG_API extern string valid;

        TS_DEBUG_API extern string auto_pad;
        TS_DEBUG_API extern string axis;
        TS_DEBUG_API extern string axes;
        TS_DEBUG_API extern string NOTSET;
        TS_DEBUG_API extern string SAME_UPPER;
        TS_DEBUG_API extern string SAME_LOWER;
        TS_DEBUG_API extern string VALID;
        TS_DEBUG_API extern string alpha;
        TS_DEBUG_API extern string beta;
        TS_DEBUG_API extern string transA;
        TS_DEBUG_API extern string transB;

        TS_DEBUG_API extern string padding_method;
        TS_DEBUG_API extern string SAME;

        TS_DEBUG_API extern string begin;
        TS_DEBUG_API extern string end;
    }

}


#endif //TENSORSTACK_BACKEDN_NAME_H
