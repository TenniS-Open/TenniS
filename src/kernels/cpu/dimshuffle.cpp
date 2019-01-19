#include <kernels/cpu/dimshuffle.h>
#include <core/tensor_builder.h>
#include <set>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

namespace ts {

//////////////////////////////////////////
Dimshuffle::Dimshuffle() {
    field(name::dim, REQUIRED);
    field(name::shuffle, REQUIRED);
    m_dim = -1;
}

void Dimshuffle::init() {
    supper::init();
 
    //TS_AUTO_CHECK(has(name::dim));
    //TS_AUTO_CHECK(has(name::shuffle));

    const Tensor& tensor_dim = get(name::dim);
    m_dim = ts::tensor::to_int(tensor_dim);
    TS_AUTO_CHECK(m_dim >= 0);   
    Tensor tensor_shuffle = tensor::cast(INT32, get(name::shuffle));
    
    m_shuffle.resize(tensor_shuffle.count());
    for(int i=0; i<tensor_shuffle.count(); i++) {
        m_shuffle[i] = tensor_shuffle.data<int>()[i];
    }
}

void Dimshuffle::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    TS_AUTO_CHECK(input_num == 1); 

    Shape shape = stack.index(0)->sizes();

    TS_AUTO_CHECK(shape.size()  > 0); 
    TS_AUTO_CHECK(m_dim >= 0 && m_dim < shape.size());

    for(int i=0; i<m_shuffle.size(); i++) {
        TS_AUTO_CHECK(m_shuffle[i] >= 0 && m_shuffle[i] < shape[m_dim]); 
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
    Tensor::Prototype output;
    infer_private(stack, output);

    stack.push(output, MemoryDevice(CPU));
    Tensor *tensor = stack.index(-1);

    Tensor *input_tensor = stack.index(0);
    const Shape& shape = input_tensor->sizes();
    const Shape& reshape = output.sizes();

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
    int type_len = type_bytes(input_tensor->dtype());
    int ncpy = backstride * type_len;

    char *psrc = input_tensor->sync(MemoryDevice(CPU)).data<char>();
    for(int k=0; k<preoffset; k++) {
        for(int i=0; i<m_shuffle.size(); i++) {
            memcpy(tensor->data<char>() + type_len * (k * newstride + i * backstride), MemoryDevice(CPU), ncpy,
                     psrc + type_len * (k * stride + m_shuffle[i] * backstride), MemoryDevice(CPU),
                     ncpy);
        }
    }

    return 1;
}








}



///////////////////////////////////////////
using namespace ts;
TS_REGISTER_OPERATOR(Dimshuffle, CPU, name::layer::dimshuffle())
