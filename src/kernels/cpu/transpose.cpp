#include <kernels/cpu/transpose.h>
#include <set>
#include <global/operator_factory.h>
#include <backend/name.h>


namespace ts {

////////////////////////////////////////////////////////////
static int stride(const std::vector<int> &shape, int start_dim){
    int index = 1;
    for(int i=start_dim; i<shape.size(); ++i) {
        index *= shape[i];
    }
    return index;
}

static int to_index(const std::vector<int> &shape, const std::vector<int> & coordinate) {
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

static std::vector<int> to_coordinate(int index, const std::vector<int> &shape) {
    std::vector<int> corrdinate(shape.size());
    int ntmp = 0;
    for(int i=0; i<shape.size(); i++) {
        ntmp = stride(shape, i+1);

        corrdinate[i] = index / ntmp + 1;
        index %= ntmp;
    }

    return corrdinate;
}

//////////////////////////////////////////////
Transpose::Transpose() {
    field("permute", OPTIONAL);
}

void Transpose::init() {
    supper::init();

    if(has("permute")){
        const Tensor& tensor_permute  = get("permute");
        if((tensor_permute.dims() != 1)) {
             throw ts::Exception("Transpose permute parameter is not match input Tensor");
        }

        if(tensor_permute.dtype() != ts::INT32) {
            throw ts::Exception("Transpose permute parameter only support INT32 dtype");
        }

        m_permute.resize(tensor_permute.count());
        std::set<int> tmpset;
        for(int i=0; i<tensor_permute.count(); i++) {
            m_permute[i] = tensor_permute.data<int>()[i];
            if(m_permute[i] < 0 ) {
                throw ts::Exception("Transpose permute parameter is invalid");
            }
            tmpset.insert(m_permute[i]);
            //reshape[i] = shape[permute[i]];

            //if(reshape[i] <= 0) {
            //    throw ts::Exception("input tensor dim invalid");
            //}
        }
        if(tmpset.size() != m_permute.size()) {
            throw::ts::Exception("Transpose permute parameter have duplicate value");
        }
    }

}



int Transpose::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 1) {
        throw ts::Exception("transpose input parameters is more than one");
    }

    const Shape& shape = stack.index(0)->sizes();
    //Shape permute;
    Shape reshape;

    if(shape.size()  < 1) {
        throw ts::Exception("Transpose input parameters dims is litter 1");
    }

    //permute.resize(shape.size());
    reshape.resize(shape.size());

    if(!has("permute")){
        m_permute.resize(shape.size()); 
        for(int j=0, i=shape.size() - 1; i>=0; i--,j++) {
            m_permute[j] = i;
            reshape[j] = shape[i];
            if(reshape[j] <= 0) {
                throw ts::Exception("transpose input tensor dim invalid");
            }
        }
    }else {
        if(m_permute.size() != shape.size()) {
            throw ts::Exception("Transpose input tensor dim invalid");
        }
        for(int i=0; i<m_permute.size(); i++) {
            if(m_permute[i] < 0 || m_permute[i] >= shape.size()) {
                throw ts::Exception("Transpose permute parameter is invalid");
            }
            reshape[i] = shape[m_permute[i]];

            if(reshape[i] <= 0) {
                throw ts::Exception("Transpose input tensor dim invalid");
            }
        }

    }

    output = ts::Tensor::Prototype(stack.index(0)->dtype(), reshape);
    //output_permute = ts::Tensor::Prototype(stack.index(0)->dtype(), m_permute);
    return 1;
}


int Transpose::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    //ts::Tensor::Prototype permute;
    output.resize(1);
    return infer_private(stack, output[0]);
}


template<typename T>
void Transpose::transpose_run(const T * psrc, int len, T* pdst,  const Shape &shape, const Shape &reshape) {
    Shape tmpshape;
    tmpshape.resize(shape.size());

    int index = 0;
    for(unsigned int i=0; i<len; i++) {
        Shape oldshape = to_coordinate(i, shape);
        for(int k=0; k<oldshape.size(); k++) {
            tmpshape[k] = oldshape[m_permute[k]];
        }

        index = to_index(reshape, tmpshape);

        if(index < 0) {
            throw ts::Exception("transpose operator failed, index is invalid");
        }
        pdst[index] = psrc[i];
    }
}

int Transpose::run(ts::Stack &stack) {
    //ts::Tensor::Prototype output_permute;
    ts::Tensor::Prototype output;
    //std::vector<ts::Tensor::Prototype> output;
    infer_private(stack, output);

    ts::Tensor *input_tensor = stack.index(0);
    const Shape& shape = input_tensor->sizes();
    //Shape permute = output_permute.sizes();
    const Shape& reshape = output.sizes();

    //int type_len = ts::type_bytes(input_tensor->dtype());

    stack.push(output, memory_device());

    ts::Tensor *tensor = stack.index(-1);
    ts::DTYPE type = stack.index(0)->dtype();

    unsigned int ncount = input_tensor->count();
    switch(type) {
        case ts::INT8: {
            const char * psrc = stack.index(0)->data<char>();
            char * pdst = tensor->sync(memory_device()).data<char>();
            transpose_run<char>(psrc, ncount, pdst, shape, reshape);
            break;
        }
        case ts::INT16: {
            const short * psrc = stack.index(0)->data<short>();
            short * pdst = tensor->sync(memory_device()).data<short>();
            transpose_run<short>(psrc, ncount, pdst, shape, reshape);
            break;
        }
        case ts::INT32: {
            const int * psrc = stack.index(0)->data<int>();
            int * pdst = tensor->sync(memory_device()).data<int>();
            transpose_run<int>(psrc, ncount, pdst, shape, reshape);
            break;
        }
        case ts::FLOAT32: {
            const float * psrc = stack.index(0)->data<float>();
            float * pdst = tensor->sync(memory_device()).data<float>();
            transpose_run<float>(psrc, ncount, pdst, shape, reshape);
            break;
        }
        case ts::FLOAT64: {
            const double * psrc = stack.index(0)->data<double>();
            double * pdst = tensor->sync(memory_device()).data<double>();
            transpose_run<double>(psrc, ncount, pdst, shape, reshape);
            break;
        }
        defalut: {
            throw ts::Exception("transpose not support this data type");
            break;
        }
    }
    return 1;
}




}



using namespace ts;
TS_REGISTER_OPERATOR(Transpose, ts::CPU, ts::name::layer::transpose())
