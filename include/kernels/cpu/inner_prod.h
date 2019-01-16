#ifndef TS_KERNELS_INNER_PROD_H
#define TS_KERNELS_INNER_PROD_H

#include <core/tensor.h>
#include <runtime/stack.h>
#include <runtime/operator.h>


namespace ts {


class Inner_Prod : public ts::Operator {
public:

    using supper = ts::Operator;
    Inner_Prod();

    virtual void init(); 

    virtual int run(ts::Stack &stack);
    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 


private:
    template<typename T>
    void compute_inner_prod(const ts::Tensor *input_tensor, const ts::Tensor *dot_tensor, ts::Tensor *output_tensor);
    void infer_private(ts::Stack &stack, ts::Tensor::Prototype &output);


};




}

#endif
