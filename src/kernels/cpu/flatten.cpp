#include <kernels/cpu/flatten.h>
#include <core/tensor_builder.h>
#include <backend/name.h>
#include <utils/assert.h>


namespace ts {

////////////////////////////////////////////////////
Flatten::Flatten() {
}

void Flatten::init() {
    supper::init();
}


int Flatten::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    int input_num = stack.size();
    TS_AUTO_CHECK(input_num == 1); 

    const Shape& shape = stack.index(0)->sizes();
    int nsize = stack.index(0)->count();
    
    TS_AUTO_CHECK(shape.size() > 0 && nsize > 0); 

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


}

using namespace ts;
TS_REGISTER_OPERATOR(Flatten, CPU, name::layer::flatten())
