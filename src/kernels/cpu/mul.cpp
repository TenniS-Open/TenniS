#include <kernels/cpu/mul.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>


namespace ts {



//////////////////////////////////////////////
Mul::Mul() {
} 

void Mul::init() {
    supper::init();

}   

void Mul::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 2) {
        throw ts::Exception("mul must have tow input parameters");
    }

    const Shape& shape = stack.index(0)->sizes();
    const Shape& mul_shape = stack.index(1)->sizes();

    if(mul_shape.size()  != shape.size() ) {
        throw ts::Exception("mul two parameters's dims is not same");
    }

    for(int i=0; i<shape.size(); i++) {
        if((shape[i] != mul_shape[i]) && (mul_shape[i] != 1)) {
            throw ts::Exception("mul two parameters's dims do not match");
        }

        if(shape[i] <= 0) {
            throw ts::Exception("mul input data shape check failed ");
        }
    }

    if(stack.index(0)->dtype() != stack.index(1)->dtype()) {
        throw ts::Exception("mul two input data type do not match");
    }
    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
    return;
}



int Mul::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    infer_private(stack, output[0]);
    return 1;
}

template<typename T>
void Mul::dimshuffle(const Shape & shape, int dim, int mul_dim, const T* src, T* dst) {
    unsigned int preoffset = 1;
    for(int i=0; i<dim; i++) {
        preoffset *= shape[i];
    }

    unsigned int stride = 1;
    for(int i=dim; i<shape.size(); i++) {
        stride *= shape[i];
    }

    unsigned int backstride = 1;
    backstride = stride / shape[dim];
    unsigned int newstride = backstride * (mul_dim + 1);

    for(int k=0; k<preoffset; k++) {
        for(int i=0; i<mul_dim+1; i++) {
            ::memcpy(dst + k * newstride + i * backstride,
                     src + k * stride,
                     backstride * sizeof(T));
        }
    }

}




template<typename T>
void Mul::compute_mul(const ts::Tensor *input_tensor, const ts::Tensor *mul_tensor, ts::Tensor *output_tensor) {
    const Shape& shape = input_tensor->sizes();
    Shape mul_shape = mul_tensor->sizes();
    const Shape& reshape = output_tensor->sizes();

    const T* psrc = input_tensor->data<T>();
    const T* pmul = mul_tensor->data<T>();
    T* pdst = output_tensor->sync(memory_device()).data<T>();

    T* tmpbuffer = NULL;
    T* dimshuffle_buffer = NULL;
    T* ptr = NULL;
    const T* pdata = NULL;

    for(int i=0; i<shape.size(); i++) {
        if((mul_shape[i] == 1) && (shape[i] > 1)) {
            if(tmpbuffer == NULL) {
                tmpbuffer = new T [ output_tensor->count()];
                dimshuffle_buffer = new T [ output_tensor->count()];
                ::memcpy(tmpbuffer, pmul, mul_tensor->count() * sizeof(T));
                ::memset(dimshuffle_buffer, 0, mul_tensor->count() * sizeof(T));
            }else {
                ptr = tmpbuffer;
                tmpbuffer = dimshuffle_buffer;
                dimshuffle_buffer = ptr;                
            }
            dimshuffle(mul_shape, i, shape[i] - mul_shape[i], tmpbuffer, dimshuffle_buffer);
            mul_shape[i] = shape[i];
        }
    }


    if(dimshuffle_buffer != NULL) {
        pdata = dimshuffle_buffer;
    }else {
        pdata = pmul;
    }

    int ncount = input_tensor->count();
    for(int i=0; i<ncount; i++) {
        pdst[i] = psrc[i] * pdata[i];
    }
    
    if(dimshuffle_buffer != NULL) {
        delete [] dimshuffle_buffer;
        delete [] tmpbuffer;
    }
}



int Mul::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    output.resize(1);
    infer_private(stack, output[0]);

    ts::Tensor *input_tensor =  stack.index(0);
    ts::Tensor *mul_tensor  = stack.index(1);

    stack.push(output[0], memory_device());
    ts::Tensor *tensor = stack.index(-1);

    ts::DTYPE type = stack.index(0)->dtype();

    switch(type) {
        case ts::INT8: {
            compute_mul<char>(input_tensor, mul_tensor, tensor);
            break;
        }
        case ts::UINT8: {
            compute_mul<unsigned char>(input_tensor, mul_tensor, tensor);
            break;
        }
        case ts::INT16: {
            compute_mul<short>(input_tensor, mul_tensor, tensor);
            break;
        }
        case ts::UINT16: {
            compute_mul<unsigned short>(input_tensor, mul_tensor, tensor);
            break;
        }
        case ts::INT32: {
            compute_mul<int>(input_tensor, mul_tensor, tensor);
            break;
        }
        case ts::UINT32: {
            compute_mul<unsigned int>(input_tensor, mul_tensor, tensor);
            break;
        }
        case ts::FLOAT32: {
            compute_mul<float>(input_tensor, mul_tensor, tensor);
            break;
        }
        case ts::FLOAT64: {
            compute_mul<double>(input_tensor, mul_tensor, tensor);
            break;
        }
        defalut: {
            throw ts::Exception("mul not support this data type");
            break;
        }
    }
    
    return 1;
}





/////////////////////////////////////////////////

TS_REGISTER_OPERATOR(Mul, ts::CPU, "mul")

}
