#ifndef TS_KERNELS_ADD_H
#define TS_KERNELS_ADD_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>



namespace ts {


class Add : public ts::Operator {
public:

    using supper = ts::Operator;
    Add();

    virtual void init(); 

    virtual int run(ts::Stack &stack);
    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 


private:
    template<typename T>
    void dimshuffle(const Shape & shape, int dim, int add_dim, const T* src, T* dst);

    template<typename T>
    void compute_add(const ts::Tensor *input_tensor, const ts::Tensor *add_tensor, ts::Tensor *output_tensor);
    void infer_private(ts::Stack &stack, ts::Tensor::Prototype &output);


};




}

#endif
