#include <kernels/cpu/reshape.h>
#include <core/tensor_builder.h>


namespace ts {

////////////////////////////////////////////////////
Reshape::Reshape() {
    field("shape", REQUIRED);
    m_index = -1;
    m_dims = 1;
}

void Reshape::init() {
    supper::init();
    
    if(!has("shape")){
        throw ts::Exception("reshape shape parameter do not find");
    }

    const Tensor& tensor_shape  = get("shape");

    if(tensor_shape.dtype() != INT32){
        throw ts::Exception("reshape shape parameter dtype is not INT32");
    }

    int ncount = tensor_shape.count();
    bool bfind = false;

    if(ncount < 1) {
        throw ts::Exception("reshape shape parameter dims must bigger than 0");
    }

    //int nindex = -1;
    //Shape reshape(ncount);
    m_shape.resize(ncount);
    //int ntotal = 1;
    for(int i=0; i<ncount; i++) {
        m_shape[i] = tensor_shape.data<int>()[i];
        //std::cout << "i:" << reshape[i] << std::endl;
        if(m_shape[i] > 0) {
            m_dims *= m_shape[i];
        }
        if(tensor_shape.data<int>()[i] <= 0) {
            if(bfind) {
                throw ts::Exception("reshape shape parameters only one less than 0 ");
            }else {
               bfind = true;
               m_index = i;
            }
        }
    }

}


int Reshape::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    int input_num = stack.size();
    if(input_num != 1) {
        throw ts::Exception("input parameters is more than one");
    }

    const Shape& shape = stack.index(0)->sizes();

    int dims = stack.index(0)->dims();
    int nsize = stack.index(0)->count();

    if(m_index >= 0) {
        m_shape[m_index] = nsize / m_dims;
        m_dims *= m_shape[m_index];
    }

    if(nsize != m_dims) {
        throw ts::Exception("reshape shape parameter is invalid");
    }

    output.resize(1);
    output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), m_shape);
    return 1;
}


int Reshape::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    infer(stack, output);
    if(output.size() < 1) {
        throw ts::Exception("Reshape infer() failed");
    }

    Tensor tensor = stack.index(0)->reshape(output[0].sizes());
    stack.push(tensor);
    return 1;
}




/////////////////////////////////////////////////////
TS_REGISTER_OPERATOR(Reshape, ts::CPU, "_reshape")

}
