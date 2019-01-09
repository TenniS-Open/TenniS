#ifndef TS_KERNELS_CONV2D_H
#define TS_KERNELS_CONV2D_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>
#include <cstring>
#include <kernels/cpu/math_cpu.h>
#include <kernels/cpu/im2col.h>
#include <string.h>
//#include <set>

namespace ts {


class Conv2d: public ts::Operator {
public:

    using supper = ts::Operator;
    Conv2d() {
        field("format", REQUIRED);
        field("padding", REQUIRED);
        field("stride", REQUIRED);
        field("dialations", REQUIRED);
        field("group", OPTIONAL);
        field("padding_value",OPTIONAL);
    }

    virtual void init() {
        supper::init();
    }


    
    int Caculate(const int height, const int width, const int kernel_h, const int kernel_w,
                 const int pad_h_top, const int pad_h_bottom, const int pad_w_left, const int pad_w_right, const int stride_h, const int stride_w, 
                 const int dilation_h, const int dilation_w,
                 int& output_h, int& output_w) {
        output_h = (height + pad_h_top + pad_h_bottom -
                        (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        output_w = (width + pad_w_left + pad_w_right -
                        (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

        return 0;
    }



    virtual int run(ts::Stack &stack) {

        ts::Tensor::Prototype output;
        Shape padding;
        std::string format;
        Shape stride;
        Shape dialations;
        int group = 1;

        int padding_value = 0;
        infer_private(stack, group, format, padding_value, padding, stride, dialations, output);

        ts::Tensor *input_tensor = stack.index(0);
        Shape shape = input_tensor->sizes();
        Shape reshape = output.sizes();

        int type_len = ts::type_bytes(input_tensor->dtype());

        stack.push(output, memory_device());
        ts::Tensor *tensor = stack.index(-1);
     
        ts::Tensor *weight_tensor = stack.index(1);
        Shape weight_shape = weight_tensor->sizes(); 

        switch(tensor->dtype()) {
        case ts::FLOAT32:
             compute_conv<float>(input_tensor, weight_tensor, tensor, shape,reshape, 
                                 weight_shape,padding,stride,dialations, padding_value);
             break;
        case ts::FLOAT64:
             compute_conv<double>(input_tensor, weight_tensor, tensor, shape,reshape, 
                                 weight_shape,padding,stride,dialations, padding_value);
             break;
        default:
            throw ts::Exception("conv2d only support FLOAT32 and FLOAT64 type");
            break;
        }
       
        /* 
        int kernel_dims = weight_shape[1] * weight_shape[2] * weight_shape[3];
        int conv_out_spatial_dim = reshape[2] * reshape[3];
        //int col_offset = kernel_dims * conv_out_spatial_dim;
        //int weight_offset = weight_shape[0] * kernel_dims;
        int output_number_offset = reshape[1] * conv_out_spatial_dim;
        int intput_number_offset = shape[1] * shape[2] * shape[3];
        int col_buffer_size = shape[1] * weight_shape[2] * weight_shape[3] * reshape[2] * reshape[3];

        float * col_buffer = new float [col_buffer_size];
        float *pinput = input_tensor->sync(memory_device()).data<float>(); 
        float *pweight = weight_tensor->sync(memory_device()).data<float>();
        float *poutput = tensor->sync(memory_device()).data<float>();
        for(int i=0; i<shape[0]; i++) {
            ::memset(col_buffer, 0, col_buffer_size * sizeof(float)); 
            im2col_cpu(pinput, shape[1], shape[2], shape[3], weight_shape[2], weight_shape[3],
                       padding[2], padding[3], stride[2], stride[3],dialations[2],dialations[3], col_buffer);
 
                        
            ts::cpu::math<float>::gemm(ts::blas::NoTrans,ts::blas::NoTrans, weight_shape[0], conv_out_spatial_dim,
                                       kernel_dims, 1.0, pweight, col_buffer, 0, poutput);
            pinput += input_number_offset;
            poutput+= output_number_offset;
        } 
      
        delete [] col_buffer; 
        */ 
        return 1;
    }

    template<typename T>
    void compute_conv(Tensor *input_tensor, Tensor *weight_tensor, Tensor *tensor, const Shape& shape, 
                      const Shape &reshape, const Shape &weight_shape, const Shape& padding, const Shape &stride,
                      const Shape &dialations, int padding_value) {

        int kernel_dims = weight_shape[1] * weight_shape[2] * weight_shape[3];
        int conv_out_spatial_dim = reshape[2] * reshape[3];
        //int col_offset = kernel_dims * conv_out_spatial_dim;
        //int weight_offset = weight_shape[0] * kernel_dims;
        int output_number_offset = reshape[1] * conv_out_spatial_dim;
        int input_number_offset = shape[1] * shape[2] * shape[3];
        int col_buffer_size = shape[1] * weight_shape[2] * weight_shape[3] * reshape[2] * reshape[3];

        
        T * col_buffer = new T [col_buffer_size];
        T *pinput = (input_tensor->sync(memory_device())).data<T>(); 
        T *pweight = weight_tensor->sync(memory_device()).data<T>();
        T *poutput = tensor->sync(memory_device()).data<T>();
        for(int i=0; i<shape[0]; i++) {
            ::memset(col_buffer, 0, col_buffer_size * sizeof(T)); 
            im2col_cpu(pinput, shape[1], shape[2], shape[3], weight_shape[2], weight_shape[3],
                       padding[2], padding[3], stride[2], stride[3],dialations[2],dialations[3], col_buffer, padding_value);
 
                        
            ts::cpu::math<T>::gemm(ts::blas::NoTrans,ts::blas::NoTrans, weight_shape[0], conv_out_spatial_dim,
                                       kernel_dims, 1.0, pweight, col_buffer, 0, poutput);
            pinput += input_number_offset;
            poutput+= output_number_offset;
        } 
      
        delete [] col_buffer; 
         
    }


    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
        Shape padding;
        std::string format;
        Shape stride;
        Shape dialations;
        int group = 1;
        int padding_value = 0;
        output.resize(1);
        return infer_private(stack, group, format,padding_value, padding, stride, dialations, output[0]);
    }

private:
    int infer_private(ts::Stack &stack, int &group, std::string &format,int &padding_value, Shape & padding, Shape & stride, Shape & dialations, ts::Tensor::Prototype &output) {
        int input_num = stack.size();
        if(input_num != 2) {
            throw ts::Exception("conv2d must have tow input parameters");
        }

        Shape shape = stack.index(0)->sizes();

        if(shape.size()  != 4 ) {
            throw ts::Exception("conv2d first parameter's dims is not 4");
        }


        Shape weight_shape = stack.index(1)->sizes();

        if(weight_shape.size()  != 4 ) {
            throw ts::Exception("conv2d second parameter's dims is not 4");
        }

        if(!has("format")){
            throw ts::Exception("conv2d format parameter do not find"); 
        }

        if(!has("padding")){
            throw ts::Exception("conv2d padding parameter do not find"); 
        }
      
        if(!has("stride")){
            throw ts::Exception("conv2d stride parameter do not find"); 
        }

        if(!has("dialations")){
            throw ts::Exception("conv2d dialations parameter do not find"); 
        }

        Tensor tensor_format = get("format");
        if(tensor_format.dims() != 1 || tensor_format.count() != 4) {
            throw ts::Exception("conv2d format parameter check failed"); 
        }

        if(tensor_format.dtype() != ts::CHAR8) {
            throw ts::Exception("conv2d format parameter CHAR8 type"); 
        }

        format = std::string(tensor_format.sync(memory_device()).data<char>(), tensor_format.count());
        if(format != "NCHW") {
            throw ts::Exception("conv2d format parameter is not supported"); 
        }

        if(shape[1] % weight_shape[1] != 0) {
            throw ts::Exception("conv2d input parameters channels check failed"); 
        }

 
        Tensor tensor_padding = get("padding");
        if(tensor_padding.dims() != 2 || tensor_padding.dtype() != ts::INT32 || tensor_padding.count() != 8) {
            throw ts::Exception("conv2d input parameter padding check failed"); 
        }

        padding.resize(8);
        for(int i=0; i<4; i++) {
            if(i==0 || i== 1) {
                if(tensor_padding.sync(memory_device()).data<int>()[2*i] != 0 || 
                   tensor_padding.sync(memory_device()).data<int>()[2*i + 1] != 0) {
                     throw ts::Exception("conv2d input parameter padding  check failed"); 
                }
            }
            padding[2*i] = tensor_padding.sync(memory_device()).data<int>()[2*i]; 
            padding[2*i + 1] = tensor_padding.sync(memory_device()).data<int>()[2*i + 1]; 
        }

        Tensor tensor_stride = get("stride");
        if(tensor_stride.dims() != 1 || tensor_stride.dtype() != ts::INT32 || tensor_stride.count() != 4) {
            throw ts::Exception("conv2d input parameter stride check failed"); 
        }

        stride.resize(4);
        for(int i=0; i<4; i++) {
            if(i==0 || i== 1) {
                if(tensor_stride.sync(memory_device()).data<int>()[i] != 0 ) {
                     throw ts::Exception("conv2d input parameter stride check failed"); 
                }
            }
            stride[i] = tensor_stride.sync(memory_device()).data<int>()[i]; 
        }

        Tensor tensor_dialations = get("dialations");
        if(tensor_dialations.dims() != 1 || tensor_dialations.dtype() != ts::INT32 || tensor_dialations.count() != 4) {
            throw ts::Exception("conv2d input parameter dialations check failed"); 
        }

        dialations.resize(4);
        for(int i=0; i<4; i++) {
            if(i==0 || i== 1) {
                if(tensor_dialations.sync(memory_device()).data<int>()[i] != 0 ) { 
                     throw ts::Exception("conv2d input parameter dialations check failed"); 
                }
            }
            dialations[i] = tensor_dialations.sync(memory_device()).data<int>()[i]; 
        }

        group = 1;
        if(has("group")){
            Tensor tensor_group = get("group");
            if(tensor_group.dims() != 1 || tensor_group.dtype() != ts::INT32 || tensor_group.count() != 1) {
                throw ts::Exception("conv2d input parameter group check failed"); 
            }
            group = tensor_group.sync(memory_device()).data<int>()[0];
        }
       
        padding_value = 0;
        if(has("padding_value")){
            Tensor tensor_padding_value = get("padding_value");
            if(tensor_padding_value.dims() != 1 || tensor_padding_value.dtype() != ts::INT32 || tensor_padding_value.count() != 1) {
                throw ts::Exception("conv2d input parameter padding_value check failed"); 
            }
            padding_value = tensor_padding_value.sync(memory_device()).data<int>()[0];
        }
        int output_h,output_w;
        Caculate(shape[2], shape[3], weight_shape[2], weight_shape[3],padding[4], padding[5], padding[6], padding[7],
                 stride[2], stride[3], dialations[2], dialations[3], output_h, output_w); 

        shape[1] = weight_shape[0];
        shape[2] = output_h;
        shape[3] = output_w;

        std::cout << "output shape:n=" << shape[0] << ",c=" << shape[1] << ",h=" << shape[2] << ",w=" << shape[3] << std::endl;
        output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
        return 1;
    }


};

}

#endif
