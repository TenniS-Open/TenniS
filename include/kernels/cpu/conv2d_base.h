#ifndef TS_KERNELS_CONV2D_BASE_H
#define TS_KERNELS_CONV2D_BASE_H


#include <core/tensor.h>
#include <runtime/stack.h>
#include <runtime/operator.h>
#include <string.h>

namespace ts {


class Conv2d_Base: public ts::Operator {
public:

    using supper = ts::Operator;
    Conv2d_Base();

    static int Caculate(const int height, const int width, const int kernel_h, const int kernel_w,
                    const int pad_h_top, const int pad_h_bottom, const int pad_w_left,
                    const int pad_w_right, const int stride_h, const int stride_w,
                    const int dilation_h, const int dilation_w,
                    int& output_h, int& output_w);

    std::function<Operator::shared(void)> GetCreator(const std::string & name);
    virtual void init(); 

    virtual int run(ts::Stack &stack) = 0; 


    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) = 0; 


protected:
    int m_group;
    int m_padding_value;
    Shape m_padding;
    Shape m_stride;
    Shape m_dialations;
    std::string m_format;
};




}

#endif
