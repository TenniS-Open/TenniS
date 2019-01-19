#include <kernels/cpu/pad.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

namespace ts {


//////////////////////////////////////////
static int stride(const std::vector<int> &shape, int start_dim){
    int index = 1;
    for(int i=start_dim; i<shape.size(); ++i) {
        index *= shape[i];
    }
    return index;
}

static int to_index(const std::vector<int> &shape, const std::vector<int> & coordinate)
{
    if(shape.size() != coordinate.size())
        return -1;

    for(int i=0; i<shape.size(); i++) {
        if(coordinate[i] > shape[i] || coordinate[i] <= 0)
            return -1;
    }

    int index = 0;
    for(int i=0; i<shape.size(); i++) {
        index += (coordinate[i] - 1) * stride(shape, i+1);
    }

    return index;

}

static std::vector<int> to_coordinate(int index, const std::vector<int> &shape)
{
    std::vector<int> corrdinate(shape.size());
    int ntmp = 0;
    for(int i=0; i<shape.size(); i++) {
        ntmp = stride(shape, i+1);

        corrdinate[i] = index / ntmp + 1;
        index %= ntmp;
    }

    return corrdinate;
}

/////////////////////////////////////////////
Pad::Pad() {
    field(name::padding_value, OPTIONAL);
    m_padding_value = 0;
}

void Pad::init() {
    supper::init();
    if(has(name::padding_value)){
        Tensor tensor_padding_value = tensor::cast(INT32, get(name::padding_value));
        m_padding_value = ts::tensor::to_int(tensor_padding_value); 
    }

}

void Pad::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    TS_AUTO_CHECK(input_num == 2); 

    const Shape &shape = stack.index(0)->sizes();
    const Shape &pad_shape = stack.index(1)->sizes();;
    Shape reshape;
        
    TS_AUTO_CHECK(shape.size()  > 0 && pad_shape.size() == 2 && pad_shape[0] == shape.size() && pad_shape[1] == 2); 

    reshape.resize(shape.size());

    Tensor padding_tensor = tensor::cast(INT32, *stack.index(1));
    const int * padding = padding_tensor.data<int>();
    for(int i=0; i<shape.size(); i++) {
        reshape[i] = shape[i] + padding[2 * i] + padding[2 * i + 1];
        TS_AUTO_CHECK(reshape[i] > 0); 
    }

    output = ts::Tensor::Prototype(stack.index(0)->dtype(), reshape);
}



int Pad::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
     output.resize(1);
     infer_private(stack, output[0]);
     return 1;
}

template <typename T>
void Pad::padding_run(const T * psrc, int len, T* pdst, const int* padding, const Shape &shape, const Shape &reshape) {
    int index = 0;
    Shape tmpshape;
    tmpshape.resize(shape.size());

    for(unsigned int i=0; i<len; i++) {
        Shape oldshape = to_coordinate(i, reshape);

        for(int k=0; k<oldshape.size(); k++) {
            tmpshape[k] = oldshape[k] - padding[2 * k];// - padding[2 * k +1];       
        }

        index = to_index(shape, tmpshape);
        if(index >= 0 ) {
            pdst[i] = psrc[index];
        }else  {
            pdst[i] = m_padding_value;
        }
    }
}


int Pad::run(ts::Stack &stack) {
    Tensor::Prototype output;
    infer_private(stack, output);

    stack.push(output, MemoryDevice(CPU));
    Tensor *tensor = stack.index(-1);
    Tensor *input_tensor = stack.index(0);
    Tensor padding_tensor = tensor::cast(INT32, *stack.index(1));
    const Shape& shape = input_tensor->sizes();
    const Shape& reshape = output.sizes();

    const int * padding = padding_tensor.data<int>();

    Shape tmpshape;
    tmpshape.resize(shape.size());

    ts::DTYPE type = stack.index(0)->dtype();
    unsigned int ncount = tensor->count();

    switch(type) {
        case ts::INT8: {
            const char * psrc = input_tensor->sync(MemoryDevice(CPU)).data<char>();
            char * pdst = tensor->data<char>();
            padding_run<char>(psrc, ncount, pdst, padding, shape, reshape);
            break;
        }
        case ts::INT16: {
            const short * psrc = input_tensor->sync(MemoryDevice(CPU)).data<short>();
            short * pdst = tensor->data<short>();
            padding_run<short>(psrc, ncount, pdst, padding, shape, reshape);
            break;
        }
        case ts::INT32: {
            const int * psrc = input_tensor->sync(MemoryDevice(CPU)).data<int>();
            int * pdst = tensor->data<int>();
            padding_run<int>(psrc, ncount, pdst, padding, shape, reshape);
            break;
        }
        case ts::FLOAT32: {
            const float * psrc = input_tensor->sync(MemoryDevice(CPU)).data<float>();
            float * pdst = tensor->data<float>();
            padding_run<float>(psrc, ncount, pdst, padding, shape, reshape);
            break;
        }
        case ts::FLOAT64: {
            const double * psrc = input_tensor->sync(MemoryDevice(CPU)).data<double>();
            double * pdst = tensor->data<double>();
            padding_run<double>(psrc, ncount, pdst, padding, shape, reshape);
            break;
        }
        defalut: {
            throw Exception("pad not support this data type");
            break;
        }
    }

    return 1;
}


}



using namespace ts;
TS_REGISTER_OPERATOR(Pad, CPU, name::layer::pad())
