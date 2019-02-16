#include <kernels/cpu/conv2d_v2.h>
#include <core/tensor_builder.h>
//#include <kernels/cpu/math_cpu.h>
//#include <kernels/cpu/im2col.h>

#include <kernels/cpu/conv2d_base.h>

#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

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
    TS_AUTO_CHECK(input_num == 3); 

    Shape shape = stack.index(0)->sizes();

    TS_AUTO_CHECK(shape.size()  == 4 ); 

    Tensor tensor_padding = tensor::cast(INT32, *stack.index(1));
    TS_AUTO_CHECK(tensor_padding.dims() == 2 && tensor_padding.count() == 8); 
   
     
    m_padding.resize(8);
    for(int i=0; i<4; i++) {
        if(i==0 || i== 1) {
            TS_AUTO_CHECK(tensor_padding.data<int>()[2*i] == 0 && 
               tensor_padding.data<int>()[2*i + 1] == 0); 
        }
        m_padding[2*i] = tensor_padding.data<int>()[2*i];
        m_padding[2*i + 1] = tensor_padding.data<int>()[2*i + 1];
    }


    const Shape& weight_shape = stack.index(2)->sizes();

    TS_AUTO_CHECK(weight_shape.size()  == 4 ); 

    //if(shape[1] % weight_shape[1] != 0) {
    //    throw ts::Exception("conv2d_v2 input parameters channels check failed");
    //}

    TS_AUTO_CHECK(stack.index(0)->dtype() == stack.index(1)->dtype());
    int output_h,output_w;
    ts::Conv2d_Base::Caculate(shape[2], shape[3], weight_shape[2], weight_shape[3],m_padding[4], m_padding[5], m_padding[6], m_padding[7],
                 m_stride[2], m_stride[3], m_dialations[2], m_dialations[3], output_h, output_w);

    shape[1] = weight_shape[0];
    shape[2] = output_h;
    shape[3] = output_w;

    output = Tensor::Prototype(stack.index(0)->dtype(), shape);
    return 1;
}


int Conv2d_V2::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    return infer_private(stack, output[0]);
}


int Conv2d_V2::run(ts::Stack &stack) {
    Tensor::Prototype output;
    infer_private(stack, output);

    Tensor *input_tensor = stack.index(0);
    Tensor *weight_tensor = stack.index(2);

    auto conv2d_creator = Conv2d_V2_Base::GetCreator(name::layer::conv2d());
    Operator::shared conv2d_op = conv2d_creator();

    conv2d_op->set(name::stride,get(name::stride));
    conv2d_op->set(name::dilation, get(name::dilation));
    conv2d_op->set(name::format,get(name::format));
    if(has(name::padding_value)) {
        conv2d_op->set(name::padding_value, get(name::padding_value));
    }

    //if(has(name::group)) {
    //    conv2d_op->set(name::group, get(name::group));
    //}

    conv2d_op->set(name::padding,ts::tensor::clone(stack.index(1)->dtype(), *stack.index(1)));

    stack.push(*input_tensor);
    stack.push(*weight_tensor);
    conv2d_op->init();
    RunOperator(conv2d_op, stack, 2);

    return 1;
}



}



using namespace ts;
TS_REGISTER_OPERATOR(Conv2d_V2, CPU, name::layer::conv2d_v2())
