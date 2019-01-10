#ifndef TS_KERNELS_DIMSHUFFLE_H
#define TS_KERNELS_DIMSHUFFLE_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>
#include <cstring>
#include <math.h>



namespace ts {


class Reshape : public ts::Operator {
public:

    using supper = ts::Operator;
    Reshape();

    virtual void init();

    virtual int run(ts::Stack &stack);

    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output);

private:
    Shape m_shape;
    int   m_index;
    int   m_dims;

};

}

#endif
