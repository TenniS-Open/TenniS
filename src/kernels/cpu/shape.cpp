#include <kernels/cpu/shape_operator.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>


namespace ts {



/////////////////////////////////////////////////////
Shape_Operator:: Shape_Operator() {
}

void Shape_Operator::init() {
    supper::init();
}

int Shape_Operator::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    int input_num = stack.size();
    TS_AUTO_CHECK(input_num == 1); 

    Shape shape;
    shape.resize(1);
    shape[0] = stack.index(0)->dims();
    output.resize(1);
    output[0] = ts::Tensor::Prototype(ts::INT32, shape);
    return 1;
}

int Shape_Operator::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    infer(stack, output);
    TS_AUTO_CHECK(output.size() > 0); 

    stack.push(output[0], MemoryDevice(CPU));
    Tensor *tensor = stack.index(-1);

    const Shape& shape = stack.index(0)->sizes();
    int *pdata = tensor->data<int>();
    for(int i=0; i<shape.size(); i++) {
        pdata[i] = shape[i];
    }

    return 1;
}





}



using namespace ts;
TS_REGISTER_OPERATOR(Shape_Operator, CPU, name::layer::shape())
