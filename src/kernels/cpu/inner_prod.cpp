#include <kernels/cpu/inner_prod.h>
#include <core/tensor_builder.h>
#include <kernels/cpu/math_cpu.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>


namespace ts {



//////////////////////////////////////////////
Inner_Prod::Inner_Prod() {
} 

void Inner_Prod::init() {
    supper::init();

}   

void Inner_Prod::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    TS_AUTO_CHECK(input_num == 2); 

    Shape shape = stack.index(0)->sizes();
    const Shape &dot_shape = stack.index(1)->sizes();
    
    TS_AUTO_CHECK(shape.size() == 2 && shape.size() == dot_shape.size() && shape[1] == dot_shape[0]); 

    TS_AUTO_CHECK(stack.index(0)->dtype() == ts::FLOAT32 || stack.index(0)->dtype() == ts::FLOAT64); 
    TS_AUTO_CHECK(stack.index(0)->dtype() == stack.index(1)->dtype());    
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
void Inner_Prod::compute_inner_prod(Tensor *input_tensor, Tensor *dot_tensor, Tensor *output_tensor) {
    const Shape& shape = input_tensor->sizes();
    const Shape& dot_shape = dot_tensor->sizes();
    const Shape& reshape = output_tensor->sizes();

    const T* psrc = input_tensor->sync(MemoryDevice(CPU)).data<T>();
    const T* pdot = dot_tensor->sync(MemoryDevice(CPU)).data<T>();
    T* pdst = output_tensor->data<T>();


    cpu::math<T>::gemm(blas::NoTrans,blas::NoTrans,shape[0],dot_shape[1],shape[1],
                           (T)1,psrc,pdot,(T)0,pdst);
    
}



int Inner_Prod::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    output.resize(1);
    infer_private(stack, output[0]);

    stack.push(output[0], MemoryDevice(CPU));
    Tensor *tensor = stack.index(-1);

    Tensor *input_tensor =  stack.index(0);
    Tensor *dot_tensor  = stack.index(1);
    DTYPE type = stack.index(0)->dtype();

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
            throw Exception("inner_prod do not support this data type");
            break;
        }
    }
    
    return 1;
}



}

using namespace ts;
TS_REGISTER_OPERATOR(Inner_Prod, CPU, name::layer::inner_prod())
