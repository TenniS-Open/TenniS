#ifndef TS_KERNELS_DIMSHUFFLE_H
#define TS_KERNELS_DIMSHUFFLE_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>
#include <cstring>
//#include <math.h>

#include <string.h>
#include <set>

namespace ts {


class Dimshuffle : public ts::Operator {
public:

    using supper = ts::Operator;
    Dimshuffle();

    virtual void init(); 

    virtual int run(ts::Stack &stack);
    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output);

private:
    void infer_private(ts::Stack &stack, ts::Tensor::Prototype &output);

private:
    int m_dim;
    Shape m_shuffle;

};

}

#endif
