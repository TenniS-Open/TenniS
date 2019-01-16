#include <kernels/cpu/conv2d_v2.h>
#include <core/tensor_builder.h>
//#include <kernels/cpu/math_cpu.h>
//#include <kernels/cpu/im2col.h>

#include <kernels/cpu/conv2d_base.h>

#include <global/operator_factory.h>
#include <backend/name.h>


namespace ts {



////////////////////////////////////////////
Conv2d_V2::Conv2d_V2() {
/*
    field("format", REQUIRED);
    //field("padding", REQUIRED);
    field("stride", REQUIRED);
    field("dialations", REQUIRED);
    field("group", OPTIONAL);
    field("padding_value",OPTIONAL);
    m_group = 1;
    m_padding_value = 0;
*/
}

void Conv2d_V2::init() {
    supper::init();


}


int Conv2d_V2::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 3) {
        throw ts::Exception("conv2d_v2 must have tow input parameters");
    }

    Shape shape = stack.index(0)->sizes();

    if(shape.size()  != 4 ) {
        throw ts::Exception("conv2d_v2 first parameter's dims is not 4");
    }

    const Tensor * tensor_padding = stack.index(1);
    if(tensor_padding->dims() != 2 || tensor_padding->dtype() != ts::INT32 || tensor_padding->count() != 8) {
        throw ts::Exception("conv2d_v2 input parameter padding check failed");
    }
;
    m_padding.resize(8);
    for(int i=0; i<4; i++) {
        if(i==0 || i== 1) {
            if(tensor_padding->data<int>()[2*i] != 0 ||
               tensor_padding->data<int>()[2*i + 1] != 0) {
                throw ts::Exception("conv2d_v2 input parameter padding  check failed");
            }
        }
        m_padding[2*i] = tensor_padding->data<int>()[2*i];
        m_padding[2*i + 1] = tensor_padding->data<int>()[2*i + 1];
    }


    const Shape& weight_shape = stack.index(2)->sizes();

    if(weight_shape.size()  != 4 ) {
        throw ts::Exception("conv2d_v2 second parameter's dims is not 4");
    }

    if(shape[1] % weight_shape[1] != 0) {
        throw ts::Exception("conv2d_v2 input parameters channels check failed");
    }

    int output_h,output_w;
    ts::Conv2d_Base::Caculate(shape[2], shape[3], weight_shape[2], weight_shape[3],m_padding[4], m_padding[5], m_padding[6], m_padding[7],
                 m_stride[2], m_stride[3], m_dialations[2], m_dialations[3], output_h, output_w);

    shape[1] = weight_shape[0];
    shape[2] = output_h;
    shape[3] = output_w;

    std::cout << "output shape:n=" << shape[0] << ",c=" << shape[1] << ",h=" << shape[2] << ",w=" << shape[3] << std::endl;
    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
    return 1;
}


int Conv2d_V2::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    return infer_private(stack, output[0]);
}


int Conv2d_V2::run(ts::Stack &stack) {
    ts::Tensor::Prototype output;
    infer_private(stack, output);

    ts::Tensor *input_tensor = stack.index(0);
    //const Shape& shape = input_tensor->sizes();
    //const Shape& reshape = output.sizes();
    //int type_len = ts::type_bytes(input_tensor->dtype());
    //stack.push(output, memory_device());
    //ts::Tensor *tensor = stack.index(-1);

    ts::Tensor *weight_tensor = stack.index(2);

    auto conv2d_creator = ts::Conv2d_V2_Base::GetCreator("conv2d");
    ts::Operator::shared conv2d_op = conv2d_creator();

    conv2d_op->set("stride",get("stride"));
    conv2d_op->set("dialations", get("dialations"));
    conv2d_op->set("format",get("format"));
    if(has("padding_value")) {
        conv2d_op->set("padding_value", get("padding_value"));
    }

    if(has("group")) {
        conv2d_op->set("group", get("group"));
    }

    conv2d_op->set("padding",ts::tensor::clone(stack.index(1)->dtype(), *stack.index(1)));

    stack.push(*input_tensor);
    stack.push(*weight_tensor);
    conv2d_op->init();
    RunOperator(conv2d_op, stack, 2);

    return 1;
}




////////////////////////////////////////////




TS_REGISTER_OPERATOR(Conv2d_V2, ts::CPU, ts::name::layer::conv2d_v2())

}
