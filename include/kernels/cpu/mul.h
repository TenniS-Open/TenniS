#ifndef TS_KERNELS_MUL_H
#define TS_KERNELS_MUL_H

#include <core/tensor.h>
#include <runtime/stack.h>
#include <runtime/operator.h>


namespace ts {


class Mul : public ts::Operator {
public:

    using supper = ts::Operator;
    Mul();

    virtual void init(); 

    virtual int run(ts::Stack &stack);
    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 


private:
    int to_index(const HypeShape &hype, const Shape & shape, const Shape &curshape);

    template<typename T>
    void compute_run(const Tensor &input_tensor, const Tensor &right_tensor,Tensor *left_tensor);

    void infer_private(ts::Stack &stack, ts::Tensor::Prototype &output);


};




}

#endif
