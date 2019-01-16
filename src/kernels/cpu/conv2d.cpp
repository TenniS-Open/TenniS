#include <kernels/cpu/conv2d.h>
#include <core/tensor_builder.h>
#include <kernels/cpu/math_cpu.h>
#include <kernels/cpu/im2col.h>
#include <global/operator_factory.h>

namespace ts {



////////////////////////////////////////////
Conv2d::Conv2d() {
/*
    field("format", REQUIRED);
    field("padding", REQUIRED);
    field("stride", REQUIRED);
    field("dialations", REQUIRED);
    field("group", OPTIONAL);
    field("padding_value",OPTIONAL);
    m_group = 1;
    m_padding_value = 0;
*/
}

void Conv2d::init() {
    supper::init();

}


int Conv2d::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 2) {
        throw ts::Exception("conv2d must have tow input parameters");
    }

    Shape shape = stack.index(0)->sizes();

    if(shape.size()  != 4 ) {
        throw ts::Exception("conv2d first parameter's dims is not 4");
    }

    const Shape& weight_shape = stack.index(1)->sizes();

    if(weight_shape.size()  != 4 ) {
        throw ts::Exception("conv2d second parameter's dims is not 4");
    }

    if(shape[1] % weight_shape[1] != 0) {
        throw ts::Exception("conv2d input parameters channels check failed");
    }

    int output_h,output_w;
    Caculate(shape[2], shape[3], weight_shape[2], weight_shape[3],m_padding[4], m_padding[5], m_padding[6], m_padding[7],
                 m_stride[2], m_stride[3], m_dialations[2], m_dialations[3], output_h, output_w);

    shape[1] = weight_shape[0];
    shape[2] = output_h;
    shape[3] = output_w;

    std::cout << "output shape:n=" << shape[0] << ",c=" << shape[1] << ",h=" << shape[2] << ",w=" << shape[3] << std::endl;
    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
    return 1;
}


int Conv2d::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    return infer_private(stack, output[0]);
}

template<typename T>
void Conv2d::compute_conv(const Tensor *input_tensor, const Tensor *weight_tensor, Tensor *tensor, const Shape& shape,
                      const Shape &reshape, const Shape &weight_shape ) {

    int kernel_dims = weight_shape[1] * weight_shape[2] * weight_shape[3];
    int conv_out_spatial_dim = reshape[2] * reshape[3];
    //int col_offset = kernel_dims * conv_out_spatial_dim;
    //int weight_offset = weight_shape[0] * kernel_dims;
    int output_number_offset = reshape[1] * conv_out_spatial_dim;
    int input_number_offset = shape[1] * shape[2] * shape[3];
    int col_buffer_size = shape[1] * weight_shape[2] * weight_shape[3] * reshape[2] * reshape[3];


    T * col_buffer = new T [col_buffer_size];
    const T *pinput = input_tensor->data<T>();
    const T *pweight = weight_tensor->data<T>();
    T *poutput = tensor->sync(memory_device()).data<T>();
    for(int i=0; i<shape[0]; i++) {
        ::memset(col_buffer, 0, col_buffer_size * sizeof(T));
        im2col_cpu(pinput, shape[1], shape[2], shape[3], weight_shape[2], weight_shape[3],
                   m_padding[2], m_padding[3], m_stride[2], m_stride[3],m_dialations[2],m_dialations[3], col_buffer, m_padding_value);


        ts::cpu::math<T>::gemm(ts::blas::NoTrans,ts::blas::NoTrans, weight_shape[0], conv_out_spatial_dim,
                               kernel_dims, 1.0, pweight, col_buffer, 0, poutput);
        pinput += input_number_offset;
        poutput+= output_number_offset;
    }

    delete [] col_buffer;

}


int Conv2d::run(ts::Stack &stack) {
    ts::Tensor::Prototype output;
    //Shape padding;
    infer_private(stack, output);

    ts::Tensor *input_tensor = stack.index(0);
    const Shape& shape = input_tensor->sizes();
    const Shape& reshape = output.sizes();

    int type_len = ts::type_bytes(input_tensor->dtype());

    stack.push(output, memory_device());
    ts::Tensor *tensor = stack.index(-1);

    ts::Tensor *weight_tensor = stack.index(1);
    const Shape& weight_shape = weight_tensor->sizes();

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
            throw ts::Exception("conv2d only support FLOAT32 and FLOAT64 type");
            break;
        }
    }

    return 1;
}




////////////////////////////////////////////




TS_REGISTER_OPERATOR(Conv2d, ts::CPU, "conv2d")

}
