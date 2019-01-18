#include <kernels/cpu/to_float.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

namespace ts {

////////////////////////////////////////////////////
To_Float::To_Float() {
}

void To_Float::init() {
    supper::init();
}


int To_Float::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    int input_num = stack.size();
    TS_AUTO_CHECK(input_num == 1); 

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


}

using namespace ts;
TS_REGISTER_OPERATOR(To_Float, CPU, name::layer::to_float())
