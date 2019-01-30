#ifndef TS_KERNELS_CONV2D_H
#define TS_KERNELS_CONV2D_H

#include <kernels/cpu/conv2d_base.h>
#include <core/tensor.h>
#include <runtime/stack.h>
#include <string.h>

namespace ts {


class Conv2d: public ts::Conv2d_Base {
public:

    using supper = ts::Conv2d_Base;
    Conv2d();

    virtual void init(); 

    virtual int run(ts::Stack &stack); 


    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 

private:
    int infer_private(ts::Stack &stack, ts::Tensor::Prototype &output); 

    template<typename T>
    void compute_conv(Tensor *input_tensor, Tensor *weight_tensor, Tensor *tensor, const Shape& shape, 
                      const Shape &reshape, const Shape &weight_shape, Stack &stack);
private:
    //int m_group;
    //int m_padding_value;
    //Shape m_padding;
    //Shape m_stride;
    //Shape m_dialations;
    //std::string m_format;
};




}

#endif
