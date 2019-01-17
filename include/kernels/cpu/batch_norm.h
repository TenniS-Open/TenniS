#ifndef TS_KERNELS_BATCH_NORM_H
#define TS_KERNELS_BATCH_NORM_H

#include <core/tensor.h>
#include <runtime/stack.h>
#include <runtime/operator.h>


namespace ts {


class Batch_Norm : public ts::Operator {
public:

    using supper = ts::Operator;
    Batch_Norm();

    virtual void init(); 

    virtual int run(ts::Stack &stack);
    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 


private:
    template<typename T>
    void compute_batch_norm(Tensor *input_tensor, Tensor *mean_tensor,
                            Tensor *variance_tensor, ts::Tensor *output_tensor);
    void infer_private(ts::Stack &stack, ts::Tensor::Prototype &output);

private:
    int m_dim;
    float m_epsilon;

};




}

#endif
