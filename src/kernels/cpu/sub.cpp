#include <kernels/cpu/sub.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>


namespace ts {



//////////////////////////////////////////////
Sub::Sub() {
} 

void Sub::init() {
    supper::init();

}   

void Sub::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 2) {
        throw ts::Exception("sub must have tow input parameters");
    }

    const Shape& shape = stack.index(0)->sizes();
    const Shape &sub_shape = stack.index(1)->sizes();

    if(sub_shape.size()  != shape.size() ) {
        throw ts::Exception("sub two parameters's dims is not same");
    }

    for(int i=0; i<shape.size(); i++) {
        if((shape[i] != sub_shape[i]) && (sub_shape[i] != 1)) {
            throw ts::Exception("sub two parameters's dims do not match");
        }

        if(shape[i] <= 0) {
            throw ts::Exception("sub input data shape check failed ");
        }
    }

    if(stack.index(0)->dtype() != stack.index(1)->dtype()) {
        throw ts::Exception("sub two input data type do not match");
    }
    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
    return;
}



int Sub::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    infer_private(stack, output[0]);
    return 1;
}

template<typename T>
void Sub::dimshuffle(const Shape & shape, int dim, int sub_dim, const T* src, T* dst) {
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
    unsigned int newstride = backstride * (sub_dim + 1);

    for(int k=0; k<preoffset; k++) {
        for(int i=0; i<sub_dim+1; i++) {
            ::memcpy(dst + k * newstride + i * backstride,
                     src + k * stride,
                     backstride * sizeof(T));
        }
    }

}




template<typename T>
void Sub::compute_sub(const ts::Tensor *input_tensor, const ts::Tensor *sub_tensor, ts::Tensor *output_tensor) {
    const Shape& shape = input_tensor->sizes();
    Shape sub_shape = sub_tensor->sizes();
    const Shape& reshape = output_tensor->sizes();

    const T* psrc = input_tensor->data<T>();
    const T* psub = sub_tensor->data<T>();
    T* pdst = output_tensor->sync(memory_device()).data<T>();

    T* tmpbuffer = NULL;
    T* dimshuffle_buffer = NULL;
    T* ptr = NULL;
    const T* pdata = NULL;

    for(int i=0; i<shape.size(); i++) {
        if((sub_shape[i] == 1) && (shape[i] > 1)) {
            if(tmpbuffer == NULL) {
                tmpbuffer = new T [ output_tensor->count()];
                dimshuffle_buffer = new T [ output_tensor->count()];
                ::memcpy(tmpbuffer, psub, sub_tensor->count() * sizeof(T));
                ::memset(dimshuffle_buffer, 0, sub_tensor->count() * sizeof(T));
            }else {
                ptr = tmpbuffer;
                tmpbuffer = dimshuffle_buffer;
                dimshuffle_buffer = ptr;                
            }
            dimshuffle(sub_shape, i, shape[i] - sub_shape[i], tmpbuffer, dimshuffle_buffer);
            sub_shape[i] = shape[i];
        }
    }


    if(dimshuffle_buffer != NULL) {
        pdata = dimshuffle_buffer;
    }else {
        pdata = psub;
    }

    int ncount = input_tensor->count();
    for(int i=0; i<ncount; i++) {
        pdst[i] = psrc[i] - pdata[i];
    }
    
    if(dimshuffle_buffer != NULL) {
        delete [] dimshuffle_buffer;
        delete [] tmpbuffer;
    }
}



int Sub::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    output.resize(1);
    infer_private(stack, output[0]);

    ts::Tensor *input_tensor =  stack.index(0);
    ts::Tensor *sub_tensor  = stack.index(1);

    stack.push(output[0], memory_device());
    ts::Tensor *tensor = stack.index(-1);

    ts::DTYPE type = stack.index(0)->dtype();

    switch(type) {
        case ts::INT8: {
            compute_sub<char>(input_tensor, sub_tensor, tensor);
            break;
        }
        case ts::UINT8: {
            compute_sub<unsigned char>(input_tensor, sub_tensor, tensor);
            break;
        }
        case ts::INT16: {
            compute_sub<short>(input_tensor, sub_tensor, tensor);
            break;
        }
        case ts::UINT16: {
            compute_sub<unsigned short>(input_tensor, sub_tensor, tensor);
            break;
        }
        case ts::INT32: {
            compute_sub<int>(input_tensor, sub_tensor, tensor);
            break;
        }
        case ts::UINT32: {
            compute_sub<unsigned int>(input_tensor, sub_tensor, tensor);
            break;
        }
        case ts::FLOAT32: {
            compute_sub<float>(input_tensor, sub_tensor, tensor);
            break;
        }
        case ts::FLOAT64: {
            compute_sub<double>(input_tensor, sub_tensor, tensor);
            break;
        }
        defalut: {
            throw ts::Exception("sub not support this data type");
            break;
        }
    }
    
    return 1;
}



}


using namespace ts;
TS_REGISTER_OPERATOR(Sub, ts::CPU, ts::name::layer::sub())
