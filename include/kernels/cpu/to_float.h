#ifndef TS_KERNELS_TO_FLOAT_H
#define TS_KERNELS_TO_FLOAT_H

#include <core/tensor.h>
#include <runtime/stack.h>
#include <runtime/operator.h>


namespace ts {


class To_Float : public ts::Operator {
public:

    using supper = ts::Operator;
    To_Float();

    virtual void init();

    virtual int run(ts::Stack &stack);

    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output);


};




}

#endif
