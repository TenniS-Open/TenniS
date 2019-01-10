#ifndef TS_KERNELS_SHAPE_OPERATOR_H
#define TS_KERNELS_SHAPE_OPERATOR_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>
//#include <cstring>
//#include <math.h>



namespace ts {


class Shape_Operator : public ts::Operator {
public:

    using supper = ts::Operator;
    Shape_Operator(); 

    virtual void init(); 

    virtual int run(ts::Stack &stack);

    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 

};


TS_REGISTER_OPERATOR(Shape_Operator, ts::CPU, "_shape")

}

#endif
