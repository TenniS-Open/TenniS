#include <kernels/cpu/sub.h>
#include <core/tensor_builder.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <global/operator_factory.h>
#include <core/device.h>

namespace ts {



//////////////////////////////////////////////
Sub::Sub() {
} 

void Sub::init() {
    supper::init();

}   

void Sub::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    TS_AUTO_CHECK(input_num == 2); 

    Shape shape = stack.index(0)->sizes();
    const Shape& right_shape = stack.index(1)->sizes();

   TS_AUTO_CHECK(shape.size() > 0);
   TS_AUTO_CHECK(right_shape.size() > 0);

    if(shape.size() != right_shape.size()) {
        TS_AUTO_CHECK((shape.size() == 1 && shape[0] == 1) || (right_shape.size() == 1 && right_shape[0] == 1));

    } 

    for(int i=0; i<shape.size(); i++) {
        TS_AUTO_CHECK(shape[i] > 0 && right_shape[i] > 0);
        if(shape[i] != right_shape[i]) {
            TS_AUTO_CHECK(shape[i] == 1 || right_shape[i] == 1); 
            if(shape[i] == 1) {
                shape[i] = right_shape[i];
            }
        }
    }

    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
    return;
}


int Sub::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    infer_private(stack, output[0]);
    return 1;
}

int Sub::to_index(const HypeShape &hype, const Shape & shape, const Shape &curshape) {
    TS_AUTO_CHECK(shape.size() == curshape.size());
    Shape tmpshape(curshape);
    for(int i=0; i<shape.size(); i++) {
        if(shape[i] == 1) {
            tmpshape[i] = 0;
        } 
    }

    return hype.to_index(tmpshape);
}


template<typename T>
void Sub::compute_run(const Tensor &input_tensor, const Tensor &right_tensor, Tensor *left_tensor) {
    const Shape& shape = input_tensor.sizes();
    Shape right_shape = right_tensor.sizes();
    const Shape& left_shape = left_tensor->sizes();

    T* pleft = left_tensor->data<T>();
    const T* pinput = input_tensor.data<T>();
    const T* pright = right_tensor.data<T>();


    HypeShape left_hype(left_shape);
    HypeShape hype(shape);
    HypeShape right_hype(right_shape);

    int ncount = right_tensor.count();
    if(input_tensor.count() == 1) {
        for(int i=0; i<ncount; i++) {
            pleft[i] = pinput[0] - pright[i];
        } 
        return;
    }

    ncount = input_tensor.count();
    if(right_tensor.count() == 1) {
        for(int i=0; i<ncount; i++) {
            pleft[i] = pinput[i] - pright[0];
        } 
        return;
    }

    ncount = left_tensor->count();
    for(int i=0; i<ncount; i++) {
        std::vector<int> tmpshape = left_hype.to_coordinate(i);
        pleft[i] = pinput[to_index(hype, shape, tmpshape)] - pright[to_index(right_hype,right_shape,tmpshape)];    
    }

}



int Sub::run(ts::Stack &stack) {
    std::vector<Tensor::Prototype> output;
    output.resize(1);
    infer_private(stack, output[0]);

    stack.push(output[0], MemoryDevice(CPU));

    Tensor *left_tensor = stack.index(-1);
    Tensor input_tensor =  tensor::cast(left_tensor->dtype(), *stack.index(0));
    Tensor right_tensor  = tensor::cast(left_tensor->dtype(), *stack.index(1));

    DTYPE type = stack.index(0)->dtype();

    switch(type) {
        case INT8: {
            compute_run<char>(input_tensor, right_tensor, left_tensor);
            break;
        }
        case UINT8: {
            compute_run<unsigned char>(input_tensor, right_tensor, left_tensor);
            break;
        }
        case INT16: {
            compute_run<short>(input_tensor, right_tensor, left_tensor);
            break;
        }
        case UINT16: {
            compute_run<unsigned short>(input_tensor, right_tensor, left_tensor);
            break;
        }
        case INT32: {
            compute_run<int>(input_tensor, right_tensor, left_tensor);
            break;
        }
        case UINT32: {
            compute_run<unsigned int>(input_tensor, right_tensor, left_tensor);
            break;
        }
        case FLOAT32: {
            compute_run<float>(input_tensor, right_tensor, left_tensor);
            break;
        }
        case FLOAT64: {
            compute_run<double>(input_tensor, right_tensor, left_tensor);
            break;
        }
        defalut: {
            throw Exception("sub not support this data type");
            break;
        }
    }
    
    return 1;
}






}

using namespace ts;
TS_REGISTER_OPERATOR(Sub, CPU, name::layer::sub())

