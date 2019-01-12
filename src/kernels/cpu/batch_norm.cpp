#include <kernels/cpu/batch_norm.h>
#include <core/tensor_builder.h>

#include <global/operator_factory.h>

namespace ts {



//////////////////////////////////////////////
Batch_Norm::Batch_Norm() {
    field("epsilon", OPTIONAL);
    field("dim", REQUIRED);
    m_dim = -1;   
    m_epsilon = 0.001;  
} 

void Batch_Norm::init() {
    supper::init();

    if(has("epsilon")){
        Tensor tensor_epsilon = get("epsilon");
        m_epsilon = ts::tensor::to_float(tensor_epsilon); 
    }

    if(has("dim")){
        Tensor tensor_dim = get("dim");
        m_dim = ts::tensor::to_int(tensor_dim);
    }
}   

void Batch_Norm::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 3) {
        throw ts::Exception("batch_norm must have three input parameters");
    }

    Shape shape = stack.index(0)->sizes();

    if(m_dim < 0 || m_dim >= shape.size() ) {
        throw ts::Exception("batch_norm dim parameter check failed");
    }
   
    Shape mean_shape = stack.index(1)->sizes();
    if(mean_shape.size()  != 1 ) {
        throw ts::Exception("batch_norm mean parameter's dims is not 1");
    }

    Shape variance_shape = stack.index(2)->sizes();
    if(variance_shape.size()  != 1 ) {
        throw ts::Exception("batch_norm variance parameter's dims is not 1");
    }

    if(variance_shape[0] != shape[m_dim] || mean_shape[0] != shape[m_dim]) {
        throw ts::Exception("batch_norm mean and variance parameter check failed");
    }

    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
    return;
}



int Batch_Norm::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    infer_private(stack, output[0]);
    return 1;
}

template<typename T>
void Batch_Norm::compute_batch_norm(ts::Tensor *input_tensor, ts::Tensor *mean_tensor, 
                                    ts::Tensor *variance_tensor, ts::Tensor *output_tensor) {
    Shape shape = input_tensor->sizes();
    int predims = 1;
    int backdims = 1;
    for(int i=0; i<m_dim; i++) {
        predims *= shape[i];
    }

    for(int i=m_dim+1; i<shape.size(); i++) {
        backdims *= shape[i];
    }

    T* psrc  = input_tensor->sync(memory_device()).data<T>();
    T* pmean = mean_tensor->sync(memory_device()).data<T>();
    T* pvariance = variance_tensor->sync(memory_device()).data<T>();
    T* pdst  = output_tensor->sync(memory_device()).data<T>();
    int stridedims = backdims * shape[m_dim];
    int offset = 0;
    //std::cout << "pre:" << predims << ",back:" << backdims << ",stride:" << stridedims << ",dim:" << m_dim << std::endl;
    for(int i=0; i<predims; i++) {
        for(int k=0; k<shape[m_dim]; k++) {
            offset = i * stridedims + k * backdims;
            for(int m=0; m<backdims; m++) {
                pdst[offset + m] = (psrc[offset + m] - pmean[k]) / (sqrt(pvariance[k] + m_epsilon));
            }
        }
    }
}



int Batch_Norm::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    output.resize(1);
    infer_private(stack, output[0]);

    ts::Tensor *input_tensor =  stack.index(0);
    ts::Tensor *mean_tensor  = stack.index(1);
    ts::Tensor *variance_tensor  = stack.index(2);
    stack.push(output[0], memory_device());
    ts::Tensor *tensor = stack.index(-1);

    switch(input_tensor->dtype()) {
        case ts::FLOAT32: {
            compute_batch_norm<float>(input_tensor, mean_tensor, variance_tensor, tensor);
            break;
        }
        case ts::FLOAT64: {
            compute_batch_norm<double>(input_tensor, mean_tensor, variance_tensor, tensor);
            break;
        }
        default: {
            throw ts::Exception("batch_norm only support FLOAT32 and FLOAT64 type");
            break;
        }
    }
    return 1;
}





/////////////////////////////////////////////////
TS_REGISTER_OPERATOR(Batch_Norm, ts::CPU, "batch_norm")





}

