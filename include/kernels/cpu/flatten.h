#ifndef TS_KERNELS_FLATTEN_H
#define TS_KERNELS_FLATTEN_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>



namespace ts {


class Flatten : public ts::Operator {
public:

    using supper = ts::Operator;
    Flatten();

    virtual void init();

    virtual int run(ts::Stack &stack);

    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output);


};




}

#endif
