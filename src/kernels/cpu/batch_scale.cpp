#include <kernels/cpu/batch_scale.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

namespace ts {



//////////////////////////////////////////////
Batch_Scale::Batch_Scale() {
    field(name::dim, REQUIRED);
    m_dim = -1;   
} 

void Batch_Scale::init() {
    supper::init();

    Tensor tensor_dim = tensor::cast(INT32, get(name::dim));
    m_dim = ts::tensor::to_int(tensor_dim);
}   

void Batch_Scale::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    TS_AUTO_CHECK(input_num == 3);

    const Shape& shape = stack.index(0)->sizes();

    TS_AUTO_CHECK(m_dim >= 0 && m_dim < shape.size() ); 
   
    const Shape& scale_shape = stack.index(1)->sizes();
    TS_AUTO_CHECK(scale_shape.size()  == 1 ); 

    const Shape& bias_shape = stack.index(2)->sizes();
    TS_AUTO_CHECK(bias_shape.size()  == 1 ); 

    TS_AUTO_CHECK(scale_shape[0] == shape[m_dim] && bias_shape[0] == shape[m_dim]); 
    TS_AUTO_CHECK(stack.index(0)->dtype() == stack.index(1)->dtype());
    TS_AUTO_CHECK(stack.index(0)->dtype() == stack.index(2)->dtype());
    output = Tensor::Prototype(stack.index(0)->dtype(), shape);
    return;
}



int Batch_Scale::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    infer_private(stack, output[0]);
    return 1;
}

template<typename T>
void Batch_Scale::compute_batch_scale(Tensor *input_tensor, Tensor *scale_tensor, 
                                      Tensor *bias_tensor, Tensor *output_tensor) {
    const Shape& shape = input_tensor->sizes();
    int predims = 1;
    int backdims = 1;
    for(int i=0; i<m_dim; i++) {
        predims *= shape[i];
    }

    for(int i=m_dim+1; i<shape.size(); i++) {
        backdims *= shape[i];
    }

    const T* psrc  = input_tensor->sync(MemoryDevice(CPU)).data<T>();
    const T* pscale = scale_tensor->sync(MemoryDevice(CPU)).data<T>();
    const T* pbias = bias_tensor->sync(MemoryDevice(CPU)).data<T>();
    T* pdst  = output_tensor->data<T>();
	std::memcpy(pdst,psrc,sizeof(T)*output_tensor->count());

    int stridedims = backdims * shape[m_dim];
    int offset = 0;
    for(int i=0; i<predims; i++) {
        for(int k=0; k<shape[m_dim]; k++) {
            offset = i * stridedims + k * backdims;
			T scale_val = pscale[k];
			T bias_val = pbias[k];
			T* pdst_temp = pdst + offset;
            for(int m=0; m<backdims; m++) {
				*pdst_temp = *pdst_temp * scale_val + bias_val;
				pdst_temp++;
                //pdst[offset + m] = psrc[offset + m] * pscale[k] + pbias[k];
            }
        }
    }
}



int Batch_Scale::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    output.resize(1);
    infer_private(stack, output[0]);

    stack.push(output[0], MemoryDevice(CPU));
    Tensor *input_tensor =  stack.index(0);
    Tensor *scale_tensor  = stack.index(1);
    Tensor *bias_tensor  =  stack.index(2);
    Tensor *tensor = stack.index(-1);

    switch(input_tensor->dtype()) {
        case FLOAT32: {
            compute_batch_scale<float>(input_tensor, scale_tensor, bias_tensor, tensor);
            break;
        }
        case FLOAT64: {
            compute_batch_scale<double>(input_tensor, scale_tensor, bias_tensor, tensor);
            break;
        }
        default: {
            throw Exception("batch_scale only support FLOAT32 and FLOAT64 type");
            break;
        }
    }
    return 1;
}



}

/////////////////////////////////////////////////
using namespace ts;
TS_REGISTER_OPERATOR(Batch_Scale, CPU, name::layer::batch_scale())

