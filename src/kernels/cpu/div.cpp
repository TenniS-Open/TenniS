#include <kernels/cpu/div.h>
#include <core/tensor_builder.h>


namespace ts {



//////////////////////////////////////////////
Div::Div() {
} 

void Div::init() {
    supper::init();

}   

void Div::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 2) {
        throw ts::Exception("div must have tow input parameters");
    }

    Shape shape = stack.index(0)->sizes();
    Shape div_shape = stack.index(1)->sizes();

    if(div_shape.size()  != shape.size() ) {
        throw ts::Exception("div two parameters's dims is not same");
    }

    for(int i=0; i<shape.size(); i++) {
        if((shape[i] != div_shape[i]) && (div_shape[i] != 1)) {
            throw ts::Exception("div two parameters's dims do not match");
        }

        if(shape[i] <= 0) {
            throw ts::Exception("div input data shape check failed ");
        }
    }

    if(stack.index(0)->dtype() != stack.index(1)->dtype()) {
        throw ts::Exception("div two input data type do not match");
    }
    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
    return;
}



int Div::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    infer_private(stack, output[0]);
    return 1;
}

template<typename T>
void Div::dimshuffle(const Shape & shape, int dim, int div_dim, const T* src, T* dst) {
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
    unsigned int newstride = backstride * (div_dim + 1);

    for(int k=0; k<preoffset; k++) {
        for(int i=0; i<div_dim+1; i++) {
            ::memcpy(dst + k * newstride + i * backstride,
                     src + k * stride,
                     backstride * sizeof(T));
        }
    }

}




template<typename T>
void Div::compute_div(ts::Tensor *input_tensor, ts::Tensor *div_tensor, ts::Tensor *output_tensor) {
    Shape shape = input_tensor->sizes();
    Shape div_shape = div_tensor->sizes();
    Shape reshape = output_tensor->sizes();

    T* psrc = input_tensor->sync(memory_device()).data<T>();
    T* pdiv = div_tensor->sync(memory_device()).data<T>();
    T* pdst = output_tensor->sync(memory_device()).data<T>();

    T* tmpbuffer = NULL;
    T* dimshuffle_buffer = NULL;
    T* ptr = NULL;

    for(int i=0; i<shape.size(); i++) {
        if((div_shape[i] == 1) && (shape[i] > 1)) {
            if(tmpbuffer == NULL) {
                tmpbuffer = new T [ output_tensor->count()];
                dimshuffle_buffer = new T [ output_tensor->count()];
                ::memcpy(tmpbuffer, pdiv, div_tensor->count() * sizeof(T));
                ::memset(dimshuffle_buffer, 0, div_tensor->count() * sizeof(T));
            }else {
                ptr = tmpbuffer;
                tmpbuffer = dimshuffle_buffer;
                dimshuffle_buffer = ptr;                
            }
            dimshuffle(div_shape, i, shape[i] - div_shape[i], tmpbuffer, dimshuffle_buffer);
            div_shape[i] = shape[i];
        }
    }


    if(dimshuffle_buffer != NULL) {
        ptr = dimshuffle_buffer;
    }else {
        ptr = pdiv;
    }

    int ncount = input_tensor->count();
    for(int i=0; i<ncount; i++) {
        if(ptr[i] == 0) {
           if(psrc[i] >= 0) {
               pdst[i] = std::numeric_limits<T>::max();
           }else {
               pdst[i] = std::numeric_limits<T>::lowest();
           }
        }else {
            pdst[i] = psrc[i] / ptr[i];
        }
    }
    
    if(dimshuffle_buffer != NULL) {
        delete [] dimshuffle_buffer;
        delete [] tmpbuffer;
    }
}



int Div::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    output.resize(1);
    infer_private(stack, output[0]);

    ts::Tensor *input_tensor =  stack.index(0);
    ts::Tensor *div_tensor  = stack.index(1);

    stack.push(output[0], memory_device());
    ts::Tensor *tensor = stack.index(-1);

    ts::DTYPE type = stack.index(0)->dtype();

    switch(type) {
        case ts::INT8: {
            compute_div<char>(input_tensor, div_tensor, tensor);
            break;
        }
        case ts::UINT8: {
            compute_div<unsigned char>(input_tensor, div_tensor, tensor);
            break;
        }
        case ts::INT16: {
            compute_div<short>(input_tensor, div_tensor, tensor);
            break;
        }
        case ts::UINT16: {
            compute_div<unsigned short>(input_tensor, div_tensor, tensor);
            break;
        }
        case ts::INT32: {
            compute_div<int>(input_tensor, div_tensor, tensor);
            break;
        }
        case ts::UINT32: {
            compute_div<unsigned int>(input_tensor, div_tensor, tensor);
            break;
        }
        case ts::FLOAT32: {
            compute_div<float>(input_tensor, div_tensor, tensor);
            break;
        }
        case ts::FLOAT64: {
            compute_div<double>(input_tensor, div_tensor, tensor);
            break;
        }
        defalut: {
            throw ts::Exception("div not support this data type");
            break;
        }
    }
    
    return 1;
}





/////////////////////////////////////////////////

TS_REGISTER_OPERATOR(Div, ts::CPU, "div")

}
