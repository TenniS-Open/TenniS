#include <kernels/cpu/conv2d_v2_base.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <utils/assert.h>
#include <backend/name.h>

namespace ts {

/*
int Conv2d_V2_Base::Caculate(const int height, const int width, const int kernel_h, const int kernel_w,
                    const int pad_h_top, const int pad_h_bottom, const int pad_w_left, 
                    const int pad_w_right, const int stride_h, const int stride_w,
                    const int dilation_h, const int dilation_w,
                    int& output_h, int& output_w) {
    output_h = floor((height + pad_h_top + pad_h_bottom -
               (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1);
    output_w = floor((width + pad_w_left + pad_w_right -
               (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1);

    return 0;
}
*/

std::function<Operator::shared(void)> Conv2d_V2_Base::GetCreator(const std::string & name) {
    auto creator = OperatorCreator::Query(computing_device().type(), name);

    if(creator == nullptr) {
        creator = OperatorCreator::Query(memory_device().type(), name);
        if(creator == nullptr) {
            creator = OperatorCreator::Query(CPU, name);
        }
    }

    TS_AUTO_CHECK_NQ(creator, nullptr);
    return creator; 
}

////////////////////////////////////////////
Conv2d_V2_Base::Conv2d_V2_Base() {
    field(name::format, REQUIRED);
    //field("padding", REQUIRED);
    field(name::stride, REQUIRED);
    field(name::dialations, REQUIRED);
    //field(name::group, OPTIONAL);
    field(name::padding_value,OPTIONAL);
    //m_group = 1;
    m_padding_value = 0;
}

void Conv2d_V2_Base::init() {
    supper::init();
  
    //TS_AUTO_CHECK(has(name::format));

    //if(!has("padding")){
    //    throw ts::Exception("conv2d_v2_base padding parameter do not find");
    //}

    //TS_AUTO_CHECK(has(name::stride));

    //TS_AUTO_CHECK(has(name::dialations));

    Tensor format = tensor::cast(CHAR8, get(name::format));
    m_format = tensor::to_string(format);
    TS_AUTO_CHECK(m_format == name::NCHW);

    /*
    const Tensor& tensor_padding = get("padding");
    if(tensor_padding.dims() != 2 || tensor_padding.dtype() != ts::INT32 || tensor_padding.count() != 8) {
        throw ts::Exception("conv2d_v2_base input parameter padding check failed");
    }
    
    m_padding.resize(8);
    for(int i=0; i<4; i++) {
        if(i==0 || i== 1) {
            if(tensor_padding.data<int>()[2*i] != 0 ||
               tensor_padding.data<int>()[2*i + 1] != 0) {
                throw ts::Exception("conv2d_v2_base input parameter padding  check failed");
            }
        }
        m_padding[2*i] = tensor_padding.data<int>()[2*i];
        m_padding[2*i + 1] = tensor_padding.data<int>()[2*i + 1];
    }
    */

    Tensor tensor_stride = tensor::cast(INT32, get(name::stride));
    TS_AUTO_CHECK(tensor_stride.dims() == 1 && tensor_stride.count() == 4);

    m_stride.resize(4);
    for(int i=0; i<4; i++) {
        if(i==0 || i== 1) {
            TS_AUTO_CHECK(tensor_stride.data<int>()[i] == 0 ); 
        }
        m_stride[i] = tensor_stride.data<int>()[i];
    }

    Tensor tensor_dialations = tensor::cast(INT32, get(name::dialations));
    TS_AUTO_CHECK(tensor_dialations.dims() == 1 && tensor_dialations.count() == 4); 

    m_dialations.resize(4);
    for(int i=0; i<4; i++) {
        if(i==0 || i== 1) {
            TS_AUTO_CHECK(tensor_dialations.data<int>()[i] == 0 ); 
        }
        m_dialations[i] = tensor_dialations.data<int>()[i];
    }

    /*
    if(has(name::group)){
        const Tensor& tensor_group = get(name::group);
        m_group = ts::tensor::to_int(tensor_group);
    }
    */

    if(has(name::padding_value)){
        Tensor tensor_padding_value = tensor::cast(INT32, get(name::padding_value));
        m_padding_value = ts::tensor::to_int(tensor_padding_value);
    }

}




////////////////////////////////////////////





}
