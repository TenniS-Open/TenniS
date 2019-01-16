#include <kernels/cpu/flatten.h>
#include <core/tensor_builder.h>
#include <backend/name.h>

namespace ts {

////////////////////////////////////////////////////
Flatten::Flatten() {
}

void Flatten::init() {
    supper::init();
}


int Flatten::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    int input_num = stack.size();
    if(input_num != 1) {
        throw ts::Exception("flatten input parameters is more than one");
    }

    const Shape& shape = stack.index(0)->sizes();
    int nsize = stack.index(0)->count();
    
    if(shape.size() < 1 || nsize <= 0) {
        throw ts::Exception("flatten input parameters check failed");
    }

    Shape flatten(2);
    flatten[0] = shape[0];
    flatten[1] = nsize / shape[0];    
    output.resize(1);
    output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), flatten);
    return 1;
}


int Flatten::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    infer(stack, output);

    Tensor tensor = stack.index(0)->reshape(output[0].sizes());
    stack.push(tensor);

    return 1;
}




/////////////////////////////////////////////////////
TS_REGISTER_OPERATOR(Flatten, ts::CPU, ts::name::layer::flatten())

}
