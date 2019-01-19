#include <kernels/cpu/reshape.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

namespace ts {

////////////////////////////////////////////////////
Reshape::Reshape() {
    field(name::shape, REQUIRED);
    m_index = -1;
    m_dims = 1;
}

void Reshape::init() {
    supper::init();
    
    //TS_AUTO_CHECK(has(name::shape));

    Tensor tensor_shape  = tensor::cast(INT32, get(name::shape));

    int ncount = tensor_shape.count();
    bool bfind = false;

    TS_AUTO_CHECK(ncount >0); 

    m_shape.resize(ncount);
    for(int i=0; i<ncount; i++) {
        m_shape[i] = tensor_shape.data<int>()[i];
        if(m_shape[i] > 0) {
            m_dims *= m_shape[i];
        }
        if(tensor_shape.data<int>()[i] <= 0) {
            if(bfind) {
                throw Exception("reshape shape parameters only one less than 0 ");
            }else {
               bfind = true;
               m_index = i;
            }
        }
    }

}


int Reshape::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    int input_num = stack.size();
    TS_AUTO_CHECK(input_num == 1); 

    const Shape& shape = stack.index(0)->sizes();

    int dims = stack.index(0)->dims();
    int nsize = stack.index(0)->count();

    if(m_index >= 0) {
        m_shape[m_index] = nsize / m_dims;
        m_dims *= m_shape[m_index];
    }

    TS_AUTO_CHECK(nsize == m_dims); 

    output.resize(1);
    output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), m_shape);
    return 1;
}


int Reshape::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    infer(stack, output);
    TS_AUTO_CHECK(output.size() > 0); 

    Tensor tensor = stack.index(0)->reshape(output[0].sizes());
    stack.push(tensor);
    return 1;
}


}

using namespace ts;
TS_REGISTER_OPERATOR(Reshape, CPU, name::layer::reshape())
