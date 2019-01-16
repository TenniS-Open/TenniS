#include <kernels/cpu/inner_prod.h>
#include <core/tensor_builder.h>
#include <kernels/cpu/math_cpu.h>
#include <global/operator_factory.h>
#include <backend/name.h>

namespace ts {



//////////////////////////////////////////////
Inner_Prod::Inner_Prod() {
} 

void Inner_Prod::init() {
    supper::init();

}   

void Inner_Prod::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 2) {
        throw ts::Exception("inner_prod must have tow input parameters");
    }

    Shape shape = stack.index(0)->sizes();
    const Shape &dot_shape = stack.index(1)->sizes();

    
    if(shape.size() != 2 || shape.size() != dot_shape.size() || shape[1] != dot_shape[0]) {
        throw ts::Exception("inner_prod two input parameters dims do not match");
    }


    if(stack.index(0)->dtype() != stack.index(1)->dtype()) {
        throw ts::Exception("inner_prod two input parameters date type do not match");
    }

    if(!(stack.index(0)->dtype() == ts::FLOAT32 || stack.index(0)->dtype() == ts::FLOAT64)) {
        throw ts::Exception("inner_prod only support FLOAT32 and FLOAT64 data types");
    }
    shape[1] = dot_shape[1];
    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
    return;
}



int Inner_Prod::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    infer_private(stack, output[0]);
    return 1;
}


template<typename T>
void Inner_Prod::compute_inner_prod(const ts::Tensor *input_tensor, const ts::Tensor *dot_tensor, ts::Tensor *output_tensor) {
    const Shape& shape = input_tensor->sizes();
    const Shape& dot_shape = dot_tensor->sizes();
    const Shape& reshape = output_tensor->sizes();

    const T* psrc = input_tensor->data<T>();
    const T* pdot = dot_tensor->data<T>();
    T* pdst = output_tensor->sync(memory_device()).data<T>();


    ts::cpu::math<T>::gemm(ts::blas::NoTrans,ts::blas::NoTrans,shape[0],dot_shape[1],shape[1],
                           (T)1,psrc,pdot,(T)0,pdst);
    
}



int Inner_Prod::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    output.resize(1);
    infer_private(stack, output[0]);

    ts::Tensor *input_tensor =  stack.index(0);
    ts::Tensor *dot_tensor  = stack.index(1);

    stack.push(output[0], memory_device());
    ts::Tensor *tensor = stack.index(-1);

    ts::DTYPE type = stack.index(0)->dtype();

    switch(type) {
/*
        case ts::INT8: {
            compute_inner_prod<char>(input_tensor, dot_tensor, tensor);
            break;
        }
        case ts::UINT8: {
            compute_inner_prod<unsigned char>(input_tensor, dot_tensor, tensor);
            break;
        }
        case ts::INT16: {
            compute_inner_prod<short>(input_tensor, dot_tensor, tensor);
            break;
        }
        case ts::UINT16: {
            compute_inner_prod<unsigned short>(input_tensor, dot_tensor, tensor);
            break;
        }
        case ts::INT32: {
            compute_inner_prod<int>(input_tensor, dot_tensor, tensor);
            break;
        }
        case ts::UINT32: {
            compute_inner_prod<unsigned int>(input_tensor, dot_tensor, tensor);
            break;
        }
*/
        case ts::FLOAT32: {
            compute_inner_prod<float>(input_tensor, dot_tensor, tensor);
            break;
        }
        case ts::FLOAT64: {
            compute_inner_prod<double>(input_tensor, dot_tensor, tensor);
            break;
        }
        defalut: {
            throw ts::Exception("inner_prod do not support this data type");
            break;
        }
    }
    
    return 1;
}





/////////////////////////////////////////////////

TS_REGISTER_OPERATOR(Inner_Prod, ts::CPU, ts::name::layer::inner_prod())

}
