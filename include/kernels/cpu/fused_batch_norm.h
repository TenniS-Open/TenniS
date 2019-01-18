#ifndef TS_KERNELS_FUSED_BATCH_NORM_H
#define TS_KERNELS_FUSED_BATCH_NORM_H

#include <core/tensor.h>
#include <runtime/stack.h>
#include <runtime/operator.h>


namespace ts {


class Fused_Batch_Norm : public ts::Operator {
public:

    using supper = ts::Operator;
    Fused_Batch_Norm();

    virtual void init(); 

    virtual int run(ts::Stack &stack);
    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 


private:
    void infer_private(ts::Stack &stack, ts::Tensor::Prototype &output);
    template<typename T>
    void compute_run(Tensor *input_tensor, Tensor *gamma_tensor,
                             Tensor *beta_tensor, Tensor *mean_tensor,
                             Tensor *variance_tensor,Tensor *output_tensor);

private:
    int m_dim;
    float m_epsilon;
};




}

#endif
