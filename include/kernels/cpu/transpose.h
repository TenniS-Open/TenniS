#ifndef TS_KERNELS_TRANSPOSE_H
#define TS_KERNELS_TRANSPOSE_H

#include <core/tensor.h>
#include <runtime/stack.h>
#include <runtime/operator.h>

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
    void transpose_run(const T * psrc, int len, T* pdst,  const Shape &shape, const Shape &reshape);

private:
    Shape m_permute;

};




}

#endif
