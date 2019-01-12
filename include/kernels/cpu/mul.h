#ifndef TS_KERNELS_MUL_H
#define TS_KERNELS_MUL_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>
//#include <cstring>
//#include <math.h>



namespace ts {


class Mul : public ts::Operator {
public:

    using supper = ts::Operator;
    Mul();

    virtual void init(); 

    virtual int run(ts::Stack &stack);
    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 


private:
    template<typename T>
    void dimshuffle(const Shape & shape, int dim, int mul_dim, const T* src, T* dst);

    template<typename T>
    void compute_mul(ts::Tensor *input_tensor, ts::Tensor *mul_tensor, ts::Tensor *output_tensor);
    void infer_private(ts::Stack &stack, ts::Tensor::Prototype &output);


};


//TS_REGISTER_OPERATOR(Mul, ts::CPU, "add")


}

#endif
