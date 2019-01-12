#ifndef TS_KERNELS_SUB_H
#define TS_KERNELS_SUB_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>



namespace ts {


class Sub : public ts::Operator {
public:

    using supper = ts::Operator;
    Sub();

    virtual void init(); 

    virtual int run(ts::Stack &stack);
    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 


private:
    template<typename T>
    void dimshuffle(const Shape & shape, int dim, int add_dim, const T* src, T* dst);

    template<typename T>
    void compute_sub(const ts::Tensor *input_tensor, const ts::Tensor *sub_tensor, ts::Tensor *output_tensor);
    void infer_private(ts::Stack &stack, ts::Tensor::Prototype &output);


};




}

#endif
