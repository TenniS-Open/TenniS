#include <kernels/cpu/to_float.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>

namespace ts {

////////////////////////////////////////////////////
To_Float::To_Float() {
}

void To_Float::init() {
    supper::init();
}


int To_Float::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    int input_num = stack.size();
    if(input_num != 1) {
        throw ts::Exception("to_float input parameters is more than one");
    }

    const Shape& shape = stack.index(0)->sizes();
    output.resize(1);
    output[0] = ts::Tensor::Prototype(ts::FLOAT32, shape);
    return 1;
}


int To_Float::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    infer(stack, output);

    auto tensor = tensor::cast(FLOAT32, *stack.index(0));
    stack.push(tensor);
    
    return 1;
}




/////////////////////////////////////////////////////
TS_REGISTER_OPERATOR(To_Float, ts::CPU, "_to_float")

}
