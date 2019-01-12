#include <kernels/cpu/depthwise_conv2d.h>
#include <core/tensor_builder.h>


namespace ts {

static int Caculate(const int height, const int width, const int kernel_h, const int kernel_w,
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


////////////////////////////////////////////
Depthwise_Conv2d::Depthwise_Conv2d() {
    field("format", REQUIRED);
    field("padding", REQUIRED);
    field("stride", REQUIRED);
    field("dialations", REQUIRED);
    field("group", OPTIONAL);
    field("padding_value",OPTIONAL);
    m_group = 1;
    m_padding_value = 0;
}

void Depthwise_Conv2d::init() {
    supper::init();
  
    if(!has("format")){
        throw ts::Exception("depthwise conv2d format parameter do not find");
    }

    if(!has("padding")){
        throw ts::Exception("depthwise conv2d padding parameter do not find");
    }

    if(!has("stride")){
        throw ts::Exception("depthwise conv2d stride parameter do not find");
    }

    if(!has("dialations")){
        throw ts::Exception("depthwise conv2d dialations parameter do not find");
    }

    m_format = ts::tensor::to_string(get("format"));
    if(m_format != "NCHW") {
       throw ts::Exception("depthwise conv2d format parameter is not supported");
    }

    Tensor tensor_padding = get("padding");
    if(tensor_padding.dims() != 2 || tensor_padding.dtype() != ts::INT32 || tensor_padding.count() != 8) {
        throw ts::Exception("depthwise input parameter padding check failed");
    }

    m_padding.resize(8);
    for(int i=0; i<4; i++) {
        if(i==0 || i== 1) {
            if(tensor_padding.sync(memory_device()).data<int>()[2*i] != 0 ||
               tensor_padding.sync(memory_device()).data<int>()[2*i + 1] != 0) {
                throw ts::Exception("depthwise input parameter padding  check failed");
            }
        }
        m_padding[2*i] = tensor_padding.sync(memory_device()).data<int>()[2*i];
        m_padding[2*i + 1] = tensor_padding.sync(memory_device()).data<int>()[2*i + 1];
    }

    Tensor tensor_stride = get("stride");
    if(tensor_stride.dims() != 1 || tensor_stride.dtype() != ts::INT32 || tensor_stride.count() != 4) {
        throw ts::Exception("depthwise input parameter stride check failed");
    }

    m_stride.resize(4);
    for(int i=0; i<4; i++) {
        if(i==0 || i== 1) {
            if(tensor_stride.sync(memory_device()).data<int>()[i] != 0 ) {
                throw ts::Exception("depthwise conv2d input parameter stride check failed");
            }
        }
        m_stride[i] = tensor_stride.sync(memory_device()).data<int>()[i];
    }

    Tensor tensor_dialations = get("dialations");
    if(tensor_dialations.dims() != 1 || tensor_dialations.dtype() != ts::INT32 || tensor_dialations.count() != 4) {
        throw ts::Exception("depthwise conv2d input parameter dialations check failed");
    }

    m_dialations.resize(4);
    for(int i=0; i<4; i++) {
        if(i==0 || i== 1) {
            if(tensor_dialations.sync(memory_device()).data<int>()[i] != 0 ) {
                throw ts::Exception("depthwiseconv2d input parameter dialations check failed");
            }
        }
        m_dialations[i] = tensor_dialations.sync(memory_device()).data<int>()[i];
    }

    if(has("group")){
        Tensor tensor_group = get("group");
        m_group = ts::tensor::to_int(tensor_group);
    }

    if(has("padding_value")){
        Tensor tensor_padding_value = get("padding_value");
        m_padding_value = ts::tensor::to_int(tensor_padding_value);
    }

}


int Depthwise_Conv2d::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 2) {
        throw ts::Exception("depthwise conv2d must have tow input parameters");
    }

    Shape shape = stack.index(0)->sizes();

    if(shape.size()  != 4 ) {
        throw ts::Exception("depthwise conv2d first parameter's dims is not 4");
    }

    Shape weight_shape = stack.index(1)->sizes();

    if(weight_shape.size()  != 4 ) {
        throw ts::Exception("depthwise conv2d second parameter's dims is not 4");
    }

    if(weight_shape[0] != 1) {
        throw ts::Exception("depthwise conv2d weight parameters first dim is not 1");
    }
    if(weight_shape[1] != shape[1]) {
        throw ts::Exception("depthwise conv2d two input parameters channels is not equal");
    }

    int output_h,output_w;
    Caculate(shape[2], shape[3], weight_shape[2], weight_shape[3],m_padding[4], m_padding[5], m_padding[6], m_padding[7],
                 m_stride[2], m_stride[3], m_dialations[2], m_dialations[3], output_h, output_w);

    //shape[1] = weight_shape[];
    shape[2] = output_h;
    shape[3] = output_w;

    std::cout << "output shape:n=" << shape[0] << ",c=" << shape[1] << ",h=" << shape[2] << ",w=" << shape[3] << std::endl;
    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
    return 1;
}


int Depthwise_Conv2d::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    return infer_private(stack, output[0]);
}

template<typename T>
void Depthwise_Conv2d::compute_conv(Tensor *input_tensor, Tensor *weight_tensor, Tensor *tensor, const Shape& shape,
                      const Shape &reshape, const Shape &weight_shape ) {

    int kernel_dims = weight_shape[1] * weight_shape[2] * weight_shape[3];
    int conv_out_spatial_dim = reshape[2] * reshape[3];
    //int col_offset = kernel_dims * conv_out_spatial_dim;
    //int weight_offset = weight_shape[0] * kernel_dims;
    int output_number_offset = reshape[1] * conv_out_spatial_dim;
    int input_number_offset = shape[1] * shape[2] * shape[3];
    int col_buffer_size = shape[1] * weight_shape[2] * weight_shape[3] * reshape[2] * reshape[3];


    T *pinput = (input_tensor->sync(memory_device())).data<T>();
    T *pweight_base = weight_tensor->sync(memory_device()).data<T>();
    T *poutput = tensor->sync(memory_device()).data<T>();
    for(int n=0; n<reshape[0]; n++) {
        for(int c=0; c < reshape[1]; c++) {
            for(int h=0; h<reshape[2]; h++) {
                for(int w=0; w<reshape[3]; w++) {
                    T * pweight = pweight_base + c * weight_shape[2] * weight_shape[3];
                    T value = 0;
                    for(int kh=0; kh<weight_shape[2]; kh++) {
                        for(int kw=0; kw<weight_shape[3]; kw++) {
                            int h_in = -m_padding[4] + h * m_stride[2] + kh * m_dialations[2];
                            int w_in = -m_padding[6] + w * m_stride[3] + kw * m_dialations[3];
                            if((h_in>=0 ) && (h_in<shape[2]) && (w_in>=0) && (w_in<shape[3])) {
                                int offset = ((n * reshape[1] + c) * shape[2] + h_in) * shape[3] + w_in;
                                value += (*pweight) * pinput[offset];
                            }
                            ++pweight;
                        }
                    }
                    *poutput++ = value;
                } 
            }
        }
    }

}


int Depthwise_Conv2d::run(ts::Stack &stack) {
    ts::Tensor::Prototype output;
    Shape padding;
    infer_private(stack, output);

    ts::Tensor *input_tensor = stack.index(0);
    Shape shape = input_tensor->sizes();
    Shape reshape = output.sizes();

    int type_len = ts::type_bytes(input_tensor->dtype());

    stack.push(output, memory_device());
    ts::Tensor *tensor = stack.index(-1);

    ts::Tensor *weight_tensor = stack.index(1);
    Shape weight_shape = weight_tensor->sizes();

    switch(tensor->dtype()) {
        case ts::FLOAT32: {
             compute_conv<float>(input_tensor, weight_tensor, tensor, shape,reshape,
                                 weight_shape);
             break;
        }
        case ts::FLOAT64: {
             compute_conv<double>(input_tensor, weight_tensor, tensor, shape,reshape,
                                 weight_shape);
             break;
        }
        default: {
            throw ts::Exception("depthwise conv2d only support FLOAT32 and FLOAT64 type");
            break;
        }
    }

    return 1;
}




////////////////////////////////////////////




TS_REGISTER_OPERATOR(Depthwise_Conv2d, ts::CPU, "depthwise_conv2d")

}
