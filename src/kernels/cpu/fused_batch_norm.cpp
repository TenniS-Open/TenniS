#include <kernels/cpu/fused_batch_norm.h>

#include <kernels/cpu/batch_norm.h>
#include <kernels/cpu/batch_scale.h>
#include <core/tensor_builder.h>

#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>
#include <vector>

namespace ts {



//////////////////////////////////////////////
Fused_Batch_Norm::Fused_Batch_Norm() {
    field(name::epsilon, OPTIONAL);
    field(name::dim, REQUIRED);
    m_epsilon = 0.001;
} 

void Fused_Batch_Norm::init() {
    supper::init();

    if(has(name::dim)){
        const Tensor& tensor_dim = get(name::dim);
        m_dim = ts::tensor::to_int(tensor_dim);

    }else {
        throw Exception("fused_batch_norm missed dim parameter ");
    }
  
    if(has(name::epsilon)){
        const Tensor& tensor_epsilon = get(name::epsilon);
        m_epsilon = ts::tensor::to_float(tensor_epsilon);
    }
}   

void Fused_Batch_Norm::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    TS_AUTO_CHECK(input_num == 5); 

    const Shape& shape = stack.index(0)->sizes();

    TS_AUTO_CHECK(m_dim >= 0 && m_dim < shape.size() ); 
   
    const Shape& gamma_shape = stack.index(1)->sizes();
    TS_AUTO_CHECK(gamma_shape.size()  == 1 ); 

    const Shape& beta_shape = stack.index(2)->sizes();
    TS_AUTO_CHECK(beta_shape.size()  == 1 ); 

    const Shape& mean_shape = stack.index(3)->sizes();
    TS_AUTO_CHECK(mean_shape.size()  == 1 ); 

    const Shape& variance_shape = stack.index(4)->sizes();
    TS_AUTO_CHECK(variance_shape.size()  == 1 ); 

    TS_AUTO_CHECK(gamma_shape[0] == shape[m_dim] && beta_shape[0] == shape[m_dim] && 
       variance_shape[0] == shape[m_dim] && mean_shape[0] == shape[m_dim]); 

    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
    return;
}



int Fused_Batch_Norm::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    infer_private(stack, output[0]);
    return 1;
}


template<typename T>
void Fused_Batch_Norm::compute_run(Tensor *input_tensor, Tensor *gamma_tensor,
                             Tensor *beta_tensor, Tensor *mean_tensor,
                             Tensor *variance_tensor,Tensor *output_tensor){

    const Shape& shape = input_tensor->sizes();
    int predims = 1;
    int backdims = 1;
    for(int i=0; i<m_dim; i++) {
        predims *= shape[i];
    }

    for(int i=m_dim+1; i<shape.size(); i++) {
        backdims *= shape[i];
    }

    T* psrc  = input_tensor->sync(MemoryDevice(CPU)).data<T>();
    const T* pgamma = gamma_tensor->sync(MemoryDevice(CPU)).data<T>();
    const T* pbeta = beta_tensor->sync(MemoryDevice(CPU)).data<T>();

    const T* pmean = mean_tensor->sync(MemoryDevice(CPU)).data<T>();
    const T* pvariance = variance_tensor->sync(MemoryDevice(CPU)).data<T>();
    T* pdst  = output_tensor->data<T>();
    int stridedims = backdims * shape[m_dim];
    int offset = 0;


    std::vector<T> vec(variance_tensor->count());
    for(int i=0; i<vec.size(); i++) {
        vec[i] = sqrt(pvariance[i] + m_epsilon);
    }

    for(int i=0; i<predims; i++) {
        for(int k=0; k<shape[m_dim]; k++) {
            offset = i * stridedims + k * backdims;
            for(int m=0; m<backdims; m++) {
                pdst[offset + m] = ((psrc[offset + m] - pmean[k]) / vec[k]) * pgamma[k] + pbeta[k];
            }
        }
    }
}

int Fused_Batch_Norm::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    output.resize(1);
    infer_private(stack, output[0]);

    stack.push(output[0], MemoryDevice(CPU));

    ts::Tensor *input_tensor =  stack.index(0);
    ts::Tensor *gamma_tensor  = stack.index(1);
    ts::Tensor *beta_tensor  =  stack.index(2);
    ts::Tensor *mean_tensor  = stack.index(3);
    ts::Tensor *variance_tensor  =  stack.index(4);
    ts::Tensor *tensor = stack.index(-1);

    switch(input_tensor->dtype()) {
        case ts::FLOAT32: {
            compute_run<float>(input_tensor, gamma_tensor, beta_tensor, 
                               mean_tensor, variance_tensor,tensor);
            break;
        }
        case ts::FLOAT64: {
            compute_run<double>(input_tensor, gamma_tensor, beta_tensor, 
                               mean_tensor, variance_tensor,tensor);
            break;
        }
        default: {
            throw Exception("fused_batch_norm only support FLOAT32 and FLOAT64 type");
            break;
        }
    }


    return 1;
}


}


using namespace ts;
TS_REGISTER_OPERATOR(Fused_Batch_Norm, CPU, name::layer::fused_batch_norm())

