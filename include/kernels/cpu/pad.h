#ifndef TS_KERNELS_PAD_H
#define TS_KERNELS_PAD_H

#include <global/operator_factory.h>
#include <core/tensor.h>

//#include <core/dtype.h>
#include <runtime/stack.h>
//#include <cstring>
//#include <math.h>

//#include <string.h>
//#include <set>

namespace ts {


class Pad : public ts::Operator {
public:

    using supper = ts::Operator;
    Pad();
    

    virtual void init();
    virtual int run(ts::Stack &stack); 
    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 

private:
    void infer_private(ts::Stack &stack, ts::Tensor::Prototype &output); 

    template <typename T>
    void padding_run(T * psrc, int len, T* pdst, int* padding, const Shape &shape, const Shape &reshape);

private:
    int m_padding_value;

};



TS_REGISTER_OPERATOR(Pad, ts::CPU, "pad")


}

#endif
