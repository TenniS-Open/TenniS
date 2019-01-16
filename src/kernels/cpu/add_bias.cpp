#include <kernels/cpu/add_bias.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>

namespace ts {



//////////////////////////////////////////////
Add_Bias::Add_Bias() {
    field("format", OPTIONAL);
    field("dim", OPTIONAL);
    m_dim = -1;     
} 

void Add_Bias::init() {
    supper::init();

    if(has("format")){
        const Tensor& tensor_format = get("format");
        std::string format = ts::tensor::to_string(tensor_format);
        if(!(format == "NCHW" || format == "NHWC")) {
            throw ts::Exception("add_bias format parameter is not supported");
        }

        int nfind = format.find("C");
        if(nfind != std::string::npos) {
            m_dim = nfind;
        }
    }

    if(has("dim")){
        const Tensor& tensor_dim = get("dim");
        m_dim = ts::tensor::to_int(tensor_dim);
    }
}   

void Add_Bias::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 2) {
        throw ts::Exception("add_bias must have tow input parameters");
    }

    const Shape & shape = stack.index(0)->sizes();

    if(shape.size()  != 4 ) {
        throw ts::Exception("add_bias first parameter's dims is not 4");
    }
   
    const Shape & bias_shape = stack.index(1)->sizes();

    if(bias_shape.size()  != 1 ) {
        throw ts::Exception("add_bias second parameter's dims is not 1");
    }

    if(m_dim < 0 || m_dim >= shape.size()) {
        throw ts::Exception("add_bias dim parameter check failed");
    }

    if(stack.index(1)->count() != shape[m_dim]) {
        throw ts::Exception("add_bias count is not match input data channels");
    }
    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
    return ;
}



int Add_Bias::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    infer_private(stack, output[0]);
    return 1;
}

template<typename T>
void Add_Bias::compute_bias(const ts::Tensor *input_tensor, const ts::Tensor *bias_tensor, ts::Tensor *output_tensor) {
    const Shape& shape = input_tensor->sizes();
    int predims = 1;
    int backdims = 1;
    for(int i=0; i<m_dim; i++) {
        predims *= shape[i];
    }

    for(int i=m_dim+1; i<shape.size(); i++) {
        backdims *= shape[i];
    }

    const T* psrc  = input_tensor->data<T>();
    const T* pbias = bias_tensor->data<T>();
    T* pdst  = output_tensor->sync(memory_device()).data<T>();
    int stridedims = backdims * shape[m_dim];
    int offset = 0;
    //std::cout << "pre:" << predims << ",back:" << backdims << ",stride:" << stridedims << ",dim:" << m_dim << std::endl;
    for(int i=0; i<predims; i++) {
        for(int k=0; k<shape[m_dim]; k++) {
            offset = i * stridedims + k * backdims;
            for(int m=0; m<backdims; m++) {
                pdst[offset + m] = pbias[k] + psrc[offset + m];
            }
        }
    }
}



int Add_Bias::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    output.resize(1);
    infer_private(stack, output[0]);

    ts::Tensor *input_tensor =  stack.index(0);
    ts::Tensor *bias_tensor  = stack.index(1);

    stack.push(output[0], memory_device());
    ts::Tensor *tensor = stack.index(-1);

    switch(input_tensor->dtype()) {

        case ts::INT8: {
            compute_bias<char>(input_tensor, bias_tensor, tensor);
            break;
        }
        case ts::UINT8: {
            compute_bias<unsigned char>(input_tensor, bias_tensor, tensor);
            break;
        }
        case ts::INT16: {
            compute_bias<short>(input_tensor, bias_tensor, tensor);
            break;
        }
        case ts::UINT16: {
            compute_bias<unsigned short>(input_tensor, bias_tensor, tensor);
            break;
        }
        case ts::INT32: {
            compute_bias<int>(input_tensor, bias_tensor, tensor);
            break;
        }
        case ts::UINT32: {
            compute_bias<unsigned int>(input_tensor, bias_tensor, tensor);
            break;
        }
        case ts::FLOAT32: {
            compute_bias<float>(input_tensor, bias_tensor, tensor);
            break;
        }
        case ts::FLOAT64: {
            compute_bias<double>(input_tensor, bias_tensor, tensor);
            break;
        }
        default: {
            throw ts::Exception("add_bias do not support data type");
            break;
        }
    }
    return 1;
}




}
/////////////////////////////////////////////////

using namespace ts;
TS_REGISTER_OPERATOR(Add_Bias, ts::CPU, ts::name::layer::add_bias())

