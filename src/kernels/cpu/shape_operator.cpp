#include <kernels/cpu/shape_operator.h>
#include <global/operator_factory.h>
#include <backend/name.h>



namespace ts {



/////////////////////////////////////////////////////
Shape_Operator:: Shape_Operator() {
}

void Shape_Operator::init() {
    supper::init();
}

int Shape_Operator::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    int input_num = stack.size();
    if(input_num != 1) {
        throw ts::Exception("shape_operator input parameters only support one");
    }

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
    if(output.size() < 1) {
        throw ts::Exception("shape_operator infer() failed");
    }

    stack.push(output[0], memory_device());
    ts::Tensor *tensor = stack.index(-1);

    const Shape& shape = stack.index(0)->sizes();
    int *pdata = tensor->sync(memory_device()).data<int>();
    for(int i=0; i<shape.size(); i++) {
        pdata[i] = shape[i];
    }

    return 1;
}





}



using namespace ts;
TS_REGISTER_OPERATOR(Shape_Operator, ts::CPU, ts::name::layer::shape())
