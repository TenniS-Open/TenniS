#include <kernels/cpu/conv2d.h>
#include <core/tensor_builder.h>
#include <kernels/cpu/math_cpu.h>
#include <kernels/cpu/im2col.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

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
    TS_AUTO_CHECK(input_num == 2); 

    Shape shape = stack.index(0)->sizes();

    TS_AUTO_CHECK(shape.size()  == 4 ); 

    const Shape& weight_shape = stack.index(1)->sizes();

    TS_AUTO_CHECK(weight_shape.size()  == 4 ); 
    TS_AUTO_CHECK(stack.index(0)->dtype() == stack.index(1)->dtype());
    //if(shape[1] % weight_shape[1] != 0) {
    //    throw ts::Exception("conv2d input parameters channels check failed");
    //}

    int output_h,output_w;
    Caculate(shape[2], shape[3], weight_shape[2], weight_shape[3],m_padding[4], m_padding[5], m_padding[6], m_padding[7],
                 m_stride[2], m_stride[3], m_dialations[2], m_dialations[3], output_h, output_w);

    shape[1] = weight_shape[0];
    shape[2] = output_h;
    shape[3] = output_w;

    output = Tensor::Prototype(stack.index(0)->dtype(), shape);
    return 1;
}


int Conv2d::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    return infer_private(stack, output[0]);
}

template<typename T>
void Conv2d::compute_conv(Tensor *input_tensor, Tensor *weight_tensor, Tensor *tensor, const Shape& shape,
                      const Shape &reshape, const Shape &weight_shape ) {

    int kernel_dims = weight_shape[1] * weight_shape[2] * weight_shape[3];
    int conv_out_spatial_dim = reshape[2] * reshape[3];
    //int col_offset = kernel_dims * conv_out_spatial_dim;
    //int weight_offset = weight_shape[0] * kernel_dims;
    int output_number_offset = reshape[1] * conv_out_spatial_dim;
    int input_number_offset = shape[1] * shape[2] * shape[3];
    int col_buffer_size = shape[1] * weight_shape[2] * weight_shape[3] * reshape[2] * reshape[3];

    Shape col_shape;
    col_shape.resize(1);
    col_shape[0] = col_buffer_size;
    Tensor col_tensor(MemoryDevice(CPU), input_tensor->dtype(), col_shape); 
    T * col_buffer = col_tensor.data<T>();//new T [col_buffer_size];

    const T *pinput = input_tensor->sync(MemoryDevice(CPU)).data<T>();
    const T *pweight = weight_tensor->sync(MemoryDevice(CPU)).data<T>();
    T *poutput = tensor->data<T>();
    for(int i=0; i<shape[0]; i++) {
        ::memset(col_buffer, 0, col_buffer_size * sizeof(T));
        im2col_cpu(pinput, shape[1], shape[2], shape[3], weight_shape[2], weight_shape[3],
                   m_padding[4], m_padding[5],m_padding[6],m_padding[7], m_stride[2], m_stride[3],m_dialations[2],m_dialations[3], col_buffer, m_padding_value);


        cpu::math<T>::gemm(ts::blas::NoTrans,ts::blas::NoTrans, weight_shape[0], conv_out_spatial_dim,
                               kernel_dims, 1.0, pweight, col_buffer, 0, poutput);
        pinput += input_number_offset;
        poutput+= output_number_offset;
    }

    //delete [] col_buffer;

}


int Conv2d::run(ts::Stack &stack) {
    Tensor::Prototype output;
    infer_private(stack, output);

    stack.push(output, MemoryDevice(CPU));
    Tensor *tensor = stack.index(-1);
    Tensor *input_tensor = stack.index(0);
    Tensor *weight_tensor = stack.index(1);
    const Shape& weight_shape = weight_tensor->sizes();
    const Shape& shape = input_tensor->sizes();
    const Shape& reshape = output.sizes();

    switch(tensor->dtype()) {
        case FLOAT32: {
             compute_conv<float>(input_tensor, weight_tensor, tensor, shape,reshape,
                                 weight_shape);
             break;
        }
        case FLOAT64: {
             compute_conv<double>(input_tensor, weight_tensor, tensor, shape,reshape,
                                 weight_shape);
             break;
        }
        default: {
            throw Exception("conv2d only support FLOAT32 and FLOAT64 type");
            break;
        }
    }

    return 1;
}




////////////////////////////////////////////


}
using namespace ts;
TS_REGISTER_OPERATOR(Conv2d, CPU, name::layer::conv2d())
