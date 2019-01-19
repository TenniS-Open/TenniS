#ifndef TS_KERNELS_SHAPE_OPERATOR_H
#define TS_KERNELS_SHAPE_OPERATOR_H

#include <core/tensor.h>
#include <runtime/stack.h>
#include <runtime/operator.h>


namespace ts {


class Shape_Operator : public ts::Operator {
public:

    using supper = ts::Operator;
    Shape_Operator(); 

    virtual void init(); 

    virtual int run(ts::Stack &stack);

    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 

};



}

#endif
