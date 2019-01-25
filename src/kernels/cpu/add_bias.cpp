#include <kernels/cpu/add_bias.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <core/device.h>

namespace ts {



//////////////////////////////////////////////
Add_Bias::Add_Bias() {
    field(name::format, OPTIONAL);
    field(name::dim, OPTIONAL);
    m_dim = -1;     
} 

void Add_Bias::init() {
    supper::init();

    if(has(name::format)){
        Tensor tensor_format = tensor::cast(CHAR8,get(name::format));
        std::string format = tensor::to_string(tensor_format);
        TS_AUTO_CHECK(format == name::NCHW || format == name::NHWC);

        int nfind = format.find("C");
        if(nfind != std::string::npos) {
            m_dim = nfind;
        }
    }

    if(has(name::dim)){
        Tensor tensor_dim = tensor::cast(INT32, get(name::dim));
        m_dim = tensor::to_int(tensor_dim);
    }

    TS_AUTO_CHECK(m_dim >= 0); 
}   

void Add_Bias::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    TS_AUTO_CHECK(input_num == 2);

    const Shape & shape = stack.index(0)->sizes();

    //TS_AUTO_CHECK(shape.size()  == 4 ); 
   
    const Shape & bias_shape = stack.index(1)->sizes();

    TS_AUTO_CHECK(bias_shape.size()  == 1 );

    TS_AUTO_CHECK(m_dim >= 0 && m_dim < shape.size()); 

    TS_AUTO_CHECK(stack.index(1)->count() == shape[m_dim]);
    output = Tensor::Prototype(stack.index(0)->dtype(), shape);
    return;
}



int Add_Bias::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    infer_private(stack, output[0]);
    return 1;
}

template<typename T>
void Add_Bias::compute_bias(Tensor *input_tensor, const Tensor *bias_tensor, Tensor *output_tensor) {
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
    T* pdst  = output_tensor->data<T>();
    const T* pbias = bias_tensor->data<T>();
    int stridedims = backdims * shape[m_dim];
    int offset = 0;
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

    stack.push(output[0], MemoryDevice(CPU));
    Tensor bias_tensor  = tensor::cast(stack.index(0)->dtype(), *stack.index(1));
    Tensor *input_tensor =  stack.index(0);

    Tensor *tensor = stack.index(-1);

    switch(input_tensor->dtype()) {

        case INT8: {
            compute_bias<char>(input_tensor, &bias_tensor, tensor);
            break;
        }
        case UINT8: {
            compute_bias<unsigned char>(input_tensor, &bias_tensor, tensor);
            break;
        }
        case INT16: {
            compute_bias<short>(input_tensor, &bias_tensor, tensor);
            break;
        }
        case UINT16: {
            compute_bias<unsigned short>(input_tensor, &bias_tensor, tensor);
            break;
        }
        case INT32: {
            compute_bias<int>(input_tensor, &bias_tensor, tensor);
            break;
        }
        case UINT32: {
            compute_bias<unsigned int>(input_tensor, &bias_tensor, tensor);
            break;
        }
        case FLOAT32: {
            compute_bias<float>(input_tensor, &bias_tensor, tensor);
            break;
        }
        case FLOAT64: {
            compute_bias<double>(input_tensor, &bias_tensor, tensor);
            break;
        }
        default: {
            throw Exception("add_bias do not support data type");
            break;
        }
    }
    return 1;
}




}
/////////////////////////////////////////////////

using namespace ts;
TS_REGISTER_OPERATOR(Add_Bias, CPU, name::layer::add_bias())

