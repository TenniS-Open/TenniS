#ifndef TS_KERNELS_TRANSPOSE_H
#define TS_KERNELS_TRANSPOSE_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>
//#include <cstring>
//#include <math.h>

//#include <string.h>
//#include <set>

namespace ts {


class Transpose : public ts::Operator {
public:

    using supper = ts::Operator;
    Transpose(); 

    virtual void init();
    virtual int run(ts::Stack &stack);
    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 

private:
    int infer_private(ts::Stack &stack, ts::Tensor::Prototype &output);

    template<typename T>
    void transpose_run(T * psrc, int len, T* pdst,  const Shape &shape, const Shape &reshape);

private:
    Shape m_permute;

};


TS_REGISTER_OPERATOR(Transpose, ts::CPU, "_transpose")


}

#endif
