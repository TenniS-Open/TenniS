#include <kernels/cpu/conv2d_v2_base.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>

namespace ts {


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


std::function<Operator::shared(void)> Conv2d_V2_Base::GetCreator(const std::string & name) {
    auto creator = ts::OperatorCreator::Query(computing_device().type(), name);

    if(creator == nullptr) {
        creator = ts::OperatorCreator::Query(memory_device().type(), name);
        if(creator == nullptr) {
            creator = ts::OperatorCreator::Query(ts::CPU, name);
        }
    }

    TS_AUTO_CHECK_NQ(creator, nullptr);
    return creator; 
}
////////////////////////////////////////////
Conv2d_V2_Base::Conv2d_V2_Base() {
    field("format", REQUIRED);
    //field("padding", REQUIRED);
    field("stride", REQUIRED);
    field("dialations", REQUIRED);
    field("group", OPTIONAL);
    field("padding_value",OPTIONAL);
    m_group = 1;
    m_padding_value = 0;
}

void Conv2d_V2_Base::init() {
    supper::init();
  
    if(!has("format")){
        throw ts::Exception("conv2d_v2_base format parameter do not find");
    }

    //if(!has("padding")){
    //    throw ts::Exception("conv2d_v2_base padding parameter do not find");
    //}

    if(!has("stride")){
        throw ts::Exception("conv2d_v2_base stride parameter do not find");
    }

    if(!has("dialations")){
        throw ts::Exception("conv2d_v2_base dialations parameter do not find");
    }

    m_format = ts::tensor::to_string(get("format"));
    if(m_format != "NCHW") {
       throw ts::Exception("conv2d_v2_base format parameter is not supported");
    }

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

    const Tensor& tensor_stride = get("stride");
    if(tensor_stride.dims() != 1 || tensor_stride.dtype() != ts::INT32 || tensor_stride.count() != 4) {
        throw ts::Exception("conv2d_v2_base input parameter stride check failed");
    }

    m_stride.resize(4);
    for(int i=0; i<4; i++) {
        if(i==0 || i== 1) {
            if(tensor_stride.data<int>()[i] != 0 ) {
                throw ts::Exception("conv2d_v2_base input parameter stride check failed");
            }
        }
        m_stride[i] = tensor_stride.data<int>()[i];
    }

    const Tensor& tensor_dialations = get("dialations");
    if(tensor_dialations.dims() != 1 || tensor_dialations.dtype() != ts::INT32 || tensor_dialations.count() != 4) {
        throw ts::Exception("conv2d_v2_base input parameter dialations check failed");
    }

    m_dialations.resize(4);
    for(int i=0; i<4; i++) {
        if(i==0 || i== 1) {
            if(tensor_dialations.data<int>()[i] != 0 ) {
                throw ts::Exception("conv2d_v2_base input parameter dialations check failed");
            }
        }
        m_dialations[i] = tensor_dialations.data<int>()[i];
    }

    if(has("group")){
        const Tensor& tensor_group = get("group");
        m_group = ts::tensor::to_int(tensor_group);
    }

    if(has("padding_value")){
        const Tensor& tensor_padding_value = get("padding_value");
        m_padding_value = ts::tensor::to_int(tensor_padding_value);
    }

}




////////////////////////////////////////////





}
