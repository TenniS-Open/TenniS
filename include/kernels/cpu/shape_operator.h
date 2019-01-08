#ifndef TS_KERNELS_SHAPE_OPERATOR_H
#define TS_KERNELS_SHAPE_OPERATOR_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>
#include <cstring>
#include <math.h>



namespace ts {


class Shape_Operator : public ts::Operator {
public:

    using supper = ts::Operator;
    Shape_Operator() {
    }

    virtual void init() {
        supper::init();
    }

    virtual int run(ts::Stack &stack) {
        std::vector<ts::Tensor::Prototype> output;
        infer(stack, output);
        if(output.size() < 1) {
            throw ts::Exception("shape_operator infer() failed");
        }

        stack.push(output[0], memory_device());
        ts::Tensor *tensor = stack.index(-1);
        
        Shape shape = stack.index(0)->sizes();
        for(int i=0; i<shape.size(); i++) {
            tensor->sync(memory_device()).data<int>()[i] = shape[i];
        }

        return 1;
    }

    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {

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



};

}

#endif
