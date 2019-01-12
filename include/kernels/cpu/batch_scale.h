#ifndef TS_KERNELS_BATCH_SCALE_H
#define TS_KERNELS_BATCH_SCALE_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>



namespace ts {


class Batch_Scale : public ts::Operator {
public:

    using supper = ts::Operator;
    Batch_Scale();

    virtual void init(); 

    virtual int run(ts::Stack &stack);
    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output); 


private:
    template<typename T>
    void compute_batch_scale(const ts::Tensor *input_tensor, const ts::Tensor *scale_tensor,
                             const ts::Tensor *bias_tensor, ts::Tensor *output_tensor);
    void infer_private(ts::Stack &stack, ts::Tensor::Prototype &output);

private:
    int m_dim;

};





}

#endif
