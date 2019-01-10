#ifndef TS_KERNELS_CONV2D_H
#define TS_KERNELS_CONV2D_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>
#include <cstring>
#include <kernels/cpu/math_cpu.h>
#include <kernels/cpu/im2col.h>
#include <string.h>
//#include <set>

namespace ts {


class Conv2d: public ts::Operator {
public:

    using supper = ts::Operator;
    Conv2d();

    virtual void init(); 

    virtual int run(ts::Stack &stack); 


    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 

private:
    int infer_private(ts::Stack &stack, ts::Tensor::Prototype &output); 

    template<typename T>
    void compute_conv(Tensor *input_tensor, Tensor *weight_tensor, Tensor *tensor, const Shape& shape, 
                      const Shape &reshape, const Shape &weight_shape);
private:
    int m_group;
    int m_padding_value;
    Shape m_padding;
    Shape m_stride;
    Shape m_dialations;
    std::string m_format;
};

}

#endif
