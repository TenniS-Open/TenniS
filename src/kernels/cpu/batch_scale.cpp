#include <kernels/cpu/batch_scale.h>
#include <core/tensor_builder.h>


namespace ts {



//////////////////////////////////////////////
Batch_Scale::Batch_Scale() {
    field("dim", REQUIRED);
    m_dim = -1;   
} 

void Batch_Scale::init() {
    supper::init();

    if(has("dim")){
        Tensor tensor_dim = get("dim");
        m_dim = ts::tensor::to_int(tensor_dim);
    }else {
        throw ts::Exception("batch_scale must set dim parameter");
    }
}   

void Batch_Scale::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 3) {
        throw ts::Exception("batch_scale must have three input parameters");
    }

    Shape shape = stack.index(0)->sizes();

    if(m_dim < 0 || m_dim >= shape.size() ) {
        throw ts::Exception("batch_scale dim parameter check failed");
    }
   
    Shape scale_shape = stack.index(1)->sizes();
    if(scale_shape.size()  != 1 ) {
        throw ts::Exception("batch_scale scale parameter's dims is not 1");
    }

    Shape bias_shape = stack.index(2)->sizes();
    if(bias_shape.size()  != 1 ) {
        throw ts::Exception("batch_scale bias parameter's dims is not 1");
    }

    if(scale_shape[0] != shape[m_dim] || bias_shape[0] != shape[m_dim]) {
        throw ts::Exception("batch_scale scale and bias parameters check failed");
    }

    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
    return;
}



int Batch_Scale::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    infer_private(stack, output[0]);
    return 1;
}

template<typename T>
void Batch_Scale::compute_batch_scale(ts::Tensor *input_tensor, ts::Tensor *scale_tensor, 
                                    ts::Tensor *bias_tensor, ts::Tensor *output_tensor) {
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
    T* pscale = scale_tensor->sync(memory_device()).data<T>();
    T* pbias = bias_tensor->sync(memory_device()).data<T>();
    T* pdst  = output_tensor->sync(memory_device()).data<T>();
    int stridedims = backdims * shape[m_dim];
    int offset = 0;
    //std::cout << "pre:" << predims << ",back:" << backdims << ",stride:" << stridedims << ",dim:" << m_dim << std::endl;
    for(int i=0; i<predims; i++) {
        for(int k=0; k<shape[m_dim]; k++) {
            offset = i * stridedims + k * backdims;
            for(int m=0; m<backdims; m++) {
                pdst[offset + m] = psrc[offset + m] * pscale[k] + pbias[k];
            }
        }
    }
}



int Batch_Scale::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    output.resize(1);
    infer_private(stack, output[0]);

    ts::Tensor *input_tensor =  stack.index(0);
    ts::Tensor *scale_tensor  = stack.index(1);
    ts::Tensor *bias_tensor  = stack.index(2);
    stack.push(output[0], memory_device());
    ts::Tensor *tensor = stack.index(-1);

    switch(input_tensor->dtype()) {
        case ts::FLOAT32: {
            compute_batch_scale<float>(input_tensor, scale_tensor, bias_tensor, tensor);
            break;
        }
        case ts::FLOAT64: {
            compute_batch_scale<double>(input_tensor, scale_tensor, bias_tensor, tensor);
            break;
        }
        default: {
            throw ts::Exception("batch_scale only support FLOAT32 and FLOAT64 type");
            break;
        }
    }
    return 1;
}





/////////////////////////////////////////////////
//TS_REGISTER_OPERATOR(Batch_Scale, ts::CPU, "batch_scale")

}
