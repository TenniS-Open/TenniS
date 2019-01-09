#include <kernels/cpu/bias.h>
#include <core/tensor_builder.h>


namespace ts {



//////////////////////////////////////////////
Bias::Bias() {
    field("format", OPTIONAL);
    field("dim", OPTIONAL);
    m_dim = -1;     
} 

void Bias::init() {
    supper::init();

    if(has("format")){
        Tensor tensor_format = get("format");
        std::string format = ts::tensor::to_string(tensor_format);
        if(!(format == "NCHW" || format == "NHWC")) {
            throw ts::Exception("bias format parameter is not supported");
        }

        int nfind = format.find("C");
        if(nfind != std::string::npos) {
            m_dim = nfind;
        }
    }

    if(has("dim")){
        Tensor tensor_dim = get("dim");
        m_dim = ts::tensor::to_int(tensor_dim);
    }
}   

void Bias::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 2) {
        throw ts::Exception("bias must have tow input parameters");
    }

    Shape shape = stack.index(0)->sizes();

    if(shape.size()  != 4 ) {
        throw ts::Exception("bias first parameter's dims is not 4");
    }
   
    Shape bias_shape = stack.index(1)->sizes();

    if(bias_shape.size()  != 1 ) {
        throw ts::Exception("bias second parameter's dims is not 1");
    }

    if(m_dim < 0 || m_dim >= shape.size()) {
        throw ts::Exception("bias dim parameter check failed");
    }

    if(stack.index(1)->count() != shape[m_dim]) {
        throw ts::Exception("bias count is not match input data channels");
    }
    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
    return ;
}



int Bias::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    infer_private(stack, output[0]);
    return 1;
}

template<typename T>
void Bias::compute_bias(ts::Tensor *input_tensor, ts::Tensor *bias_tensor, ts::Tensor *output_tensor) {
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
    T* pbias = bias_tensor->sync(memory_device()).data<T>();
    T* pdst  = output_tensor->sync(memory_device()).data<T>();
    int stridedims = backdims * shape[m_dim];

    //std::cout << "pre:" << predims << ",back:" << backdims << ",stride:" << stridedims << ",dim:" << m_dim << std::endl;
    for(int i=0; i<predims; i++) {
        for(int k=0; k<shape[m_dim]; k++) {
            for(int m=0; m<backdims; m++) {
                pdst[i * stridedims + k * backdims + m] = pbias[k] + psrc[i * stridedims + k * backdims + m];
            }
        }
    }
}



int Bias::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    output.resize(1);
    infer_private(stack, output[0]);

    ts::Tensor *input_tensor =  stack.index(0);
    ts::Tensor *bias_tensor  = stack.index(1);

    stack.push(output[0], memory_device());
    ts::Tensor *tensor = stack.index(-1);

    switch(input_tensor->dtype()) {
        case ts::FLOAT32: {
            compute_bias<float>(input_tensor, bias_tensor, tensor);
            break;
        }
        case ts::FLOAT64: {
            compute_bias<double>(input_tensor, bias_tensor, tensor);
            break;
        }
        default: {
            throw ts::Exception("bias only support FLOAT32 and FLOAT64 type");
            break;
        }
    }
    return 1;
}





/////////////////////////////////////////////////
TS_REGISTER_OPERATOR(Bias, ts::CPU, "bias")

}
