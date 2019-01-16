#include <kernels/cpu/dimshuffle.h>
#include <core/tensor_builder.h>
#include <set>
#include <global/operator_factory.h>


namespace ts {

//////////////////////////////////////////
Dimshuffle::Dimshuffle() {
    field("dim", REQUIRED);
    field("shuffle", REQUIRED);
    m_dim = -1;
}

void Dimshuffle::init() {
    supper::init();
 
    if(!has("dim")){
        throw ts::Exception("Dimshuffle input parameter dim do not find");
    }

    if(!has("shuffle")){
        throw ts::Exception("Dimshuffle input parameter shuffle do not find");
    }

    const Tensor& tensor_dim = get("dim");
    m_dim = ts::tensor::to_int(tensor_dim);

    const Tensor& tensor_shuffle = get("shuffle");
    if(tensor_shuffle.dtype() != ts::INT32) {
        throw ts::Exception("Dimshuffle input parameter shuffle only support INT32 type");
    }
    m_shuffle.resize(tensor_shuffle.count());
    for(int i=0; i<tensor_shuffle.count(); i++) {
        m_shuffle[i] = tensor_shuffle.data<int>()[i];
    }
}

void Dimshuffle::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 1) {
        throw ts::Exception("Dimshuffle input parameters is more than one");
    }

    Shape shape = stack.index(0)->sizes();

    if(shape.size()  < 1) {
        throw ts::Exception("Dimshuffle input parameters dims is litter 1");
    }

    if(m_dim < 0 || m_dim >= shape.size()) {
        throw ts::Exception("Dimshuffle input parameter dim value check failed");
    }

    for(int i=0; i<m_shuffle.size(); i++) {
        if(m_shuffle[i] < 0 || m_shuffle[i] >= shape[m_dim]) {
            throw ts::Exception("Dimshuffle shuffle parameter is invalid");
        }
    }

    if(m_shuffle.size() != shape[m_dim]) {
        shape[m_dim] = m_shuffle.size();
    }
    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
}



int Dimshuffle::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    infer_private(stack, output[0]);
    return 1;
}

int Dimshuffle::run(ts::Stack &stack) {
    ts::Tensor::Prototype output;
    infer_private(stack, output);

    ts::Tensor *input_tensor = stack.index(0);
    const Shape& shape = input_tensor->sizes();
    const Shape& reshape = output.sizes();

    int type_len = ts::type_bytes(input_tensor->dtype());

    stack.push(output, memory_device());
    ts::Tensor *tensor = stack.index(-1);

    unsigned int preoffset = 1;
    for(int i=0; i<m_dim; i++) {
        preoffset *= shape[i];
    }

    unsigned int stride = 1;
    for(int i=m_dim; i<shape.size(); i++) {
        stride *= shape[i];
    }

    unsigned int backstride = 1;
    backstride = stride / shape[m_dim];
    unsigned int newstride = backstride * m_shuffle.size();

    for(int k=0; k<preoffset; k++) {
        for(int i=0; i<m_shuffle.size(); i++) {
            ::memcpy(tensor->sync(memory_device()).data<char>() + type_len * (k * newstride + i * backstride),
                     input_tensor->sync(memory_device()).data<char>() + type_len * (k * stride + m_shuffle[i] * backstride),
                     backstride * type_len);
        }
    }

    return 1;
}












///////////////////////////////////////////

TS_REGISTER_OPERATOR(Dimshuffle, ts::CPU, "_dimshuffle")

}
