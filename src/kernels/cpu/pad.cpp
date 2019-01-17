#include <kernels/cpu/pad.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>


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
    field("padding_value", OPTIONAL);
    m_padding_value = 0;
}

void Pad::init() {
    supper::init();
    if(has("padding_value")){
        const Tensor& tensor_padding_value = get("padding_value");
        m_padding_value = ts::tensor::to_int(tensor_padding_value); 
    }

}

void Pad::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 2) {
        throw ts::Exception("pad only support two input parameters");
    }

    const Shape &shape = stack.index(0)->sizes();
    const Shape &pad_shape = stack.index(1)->sizes();;
    Shape reshape;
        
    if(!(stack.index(0)->dtype() == ts::INT32 || stack.index(0)->dtype() == ts::FLOAT32 ||
        stack.index(0)->dtype() == ts::FLOAT64 || stack.index(0)->dtype() == ts::INT32 ||
        stack.index(0)->dtype() == ts::INT8 || stack.index(0)->dtype() == ts::INT16)) {
        throw ts::Exception("pad the first input parameter type is not supportted");
    }

    if(stack.index(1)->dtype() != ts::INT32) {
        throw ts::Exception("pad the second input parameter type error");
    }

    if(shape.size()  < 1 || pad_shape.size() != 2 || pad_shape[0] != shape.size() || pad_shape[1] != 2) {
        throw ts::Exception("pad input parameters dims check failed");
    }

    reshape.resize(shape.size());

    const int * padding = stack.index(1)->data<int>();
    //std::cout << "reshape:";
    for(int i=0; i<shape.size(); i++) {
        reshape[i] = shape[i] + padding[2 * i] + padding[2 * i + 1];
        if(reshape[i] <= 0) {
            throw ts::Exception("pad padding parameter value check failed");
        }

        //std::cout << reshape[i] << ",";
    }
    //std::cout << std::endl;

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
    ts::Tensor::Prototype output;
    infer_private(stack, output);

    ts::Tensor *input_tensor = stack.index(0);
    const Shape& shape = input_tensor->sizes();
    const Shape& reshape = output.sizes();

    int type_len = ts::type_bytes(input_tensor->dtype());

    stack.push(output, memory_device());
    ts::Tensor *tensor = stack.index(-1);

    const int * padding = stack.index(1)->data<int>();

    Shape tmpshape;
    tmpshape.resize(shape.size());

    ts::DTYPE type = stack.index(0)->dtype();
    unsigned int ncount = tensor->count();

    switch(type) {
        case ts::INT8: {
            const char * psrc = stack.index(0)->data<char>();
            char * pdst = tensor->sync(memory_device()).data<char>();
            padding_run<char>(psrc, ncount, pdst, padding, shape, reshape);
            break;
        }
        case ts::INT16: {
            const short * psrc = stack.index(0)->data<short>();
            short * pdst = tensor->sync(memory_device()).data<short>();
            padding_run<short>(psrc, ncount, pdst, padding, shape, reshape);
            break;
        }
        case ts::INT32: {
            const int * psrc = stack.index(0)->data<int>();
            int * pdst = tensor->sync(memory_device()).data<int>();
            padding_run<int>(psrc, ncount, pdst, padding, shape, reshape);
            break;
        }
        case ts::FLOAT32: {
            const float * psrc = stack.index(0)->data<float>();
            float * pdst = tensor->sync(memory_device()).data<float>();
            padding_run<float>(psrc, ncount, pdst, padding, shape, reshape);
            break;
        }
        case ts::FLOAT64: {
            const double * psrc = stack.index(0)->data<double>();
            double * pdst = tensor->sync(memory_device()).data<double>();
            padding_run<double>(psrc, ncount, pdst, padding, shape, reshape);
            break;
        }
        defalut: {
            throw ts::Exception("pad not support this data type");
            break;
        }
    }

    return 1;
}


}



using namespace ts;
TS_REGISTER_OPERATOR(Pad, ts::CPU, ts::name::layer::pad())
