#include <kernels/cpu/transpose.h>
#include <set>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>
#include <core/tensor_builder.h>

namespace ts {

/*
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
*/

//////////////////////////////////////////////
Transpose::Transpose() {
    field(name::permute, OPTIONAL);
}

void Transpose::init() {
    supper::init();

    if(has(name::permute)){
        Tensor tensor_permute = tensor::cast(INT32, get(name::permute));
        TS_AUTO_CHECK(tensor_permute.dims() == 1);

        m_permute.resize(tensor_permute.count());
        std::set<int> tmpset;
        for(int i=0; i<tensor_permute.count(); i++) {
            m_permute[i] = tensor_permute.data<int>()[i];
            TS_AUTO_CHECK(m_permute[i] >= 0 ); 
            tmpset.insert(m_permute[i]);
            //reshape[i] = shape[permute[i]];

            //if(reshape[i] <= 0) {
            //    throw ts::Exception("input tensor dim invalid");
            //}
        }
        if(tmpset.size() != m_permute.size()) {
            throw Exception("Transpose permute parameter have duplicate value");
        }
    }

}



int Transpose::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    TS_AUTO_CHECK(input_num == 1); 

    const Shape& shape = stack.index(0)->sizes();
    Shape reshape;

    TS_AUTO_CHECK(shape.size() > 0); 
    reshape.resize(shape.size());

    if(!has(name::permute)){
        m_permute = shape; 
        if(m_permute.size() >= 2) {
            m_permute[shape.size() - 2] = shape[shape.size() - 1];
            m_permute[shape.size() - 1] = shape[shape.size() - 2];
        }
        /*
        for(int j=0, i=shape.size() - 1; i>=0; i--,j++) {
            m_permute[j] = i;
            reshape[j] = shape[i];
            TS_AUTO_CHECK(reshape[j] > 0); 
        }
        */
    }else {
        TS_AUTO_CHECK(m_permute.size() == shape.size()); 
        for(int i=0; i<m_permute.size(); i++) {
            TS_AUTO_CHECK(m_permute[i] >= 0 && m_permute[i] < shape.size()); 
            reshape[i] = shape[m_permute[i]];

            TS_AUTO_CHECK(reshape[i] > 0); 
        }

    }

    output = ts::Tensor::Prototype(stack.index(0)->dtype(), reshape);
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

    HypeShape hype_shape(shape);
    HypeShape hype_reshape(reshape);

    int index = 0;
    for(unsigned int i=0; i<len; i++) {
        Shape oldshape = hype_shape.to_coordinate(i);//to_coordinate(i, shape);
        for(int k=0; k<oldshape.size(); k++) {
            tmpshape[k] = oldshape[m_permute[k]];
        }

        index = hype_reshape.to_index(tmpshape);//to_index(reshape, tmpshape);

        //if(index < 0) {
        //    throw ts::Exception("transpose operator failed, index is invalid");
        //}
        pdst[index] = psrc[i];
    }
}

int Transpose::run(ts::Stack &stack) {
    ts::Tensor::Prototype output;
    infer_private(stack, output);

    stack.push(output, MemoryDevice(CPU));
    ts::Tensor *tensor = stack.index(-1);
    ts::Tensor *input_tensor = stack.index(0);
    const Shape& shape = input_tensor->sizes();
    const Shape& reshape = output.sizes();

    ts::DTYPE type = stack.index(0)->dtype();
    unsigned int ncount = input_tensor->count();
    switch(type) {
        case ts::INT8: {
            const char * psrc = stack.index(0)->sync(MemoryDevice(CPU)).data<char>();
            char * pdst = tensor->data<char>();
            transpose_run<char>(psrc, ncount, pdst, shape, reshape);
            break;
        }
        case ts::INT16: {
            const short * psrc = stack.index(0)->sync(MemoryDevice(CPU)).data<short>();
            short * pdst = tensor->data<short>();
            transpose_run<short>(psrc, ncount, pdst, shape, reshape);
            break;
        }
        case ts::INT32: {
            const int * psrc = stack.index(0)->sync(MemoryDevice(CPU)).data<int>();
            int * pdst = tensor->data<int>();
            transpose_run<int>(psrc, ncount, pdst, shape, reshape);
            break;
        }
        case ts::FLOAT32: {
            const float * psrc = stack.index(0)->sync(MemoryDevice(CPU)).data<float>();
            float * pdst = tensor->data<float>();
            transpose_run<float>(psrc, ncount, pdst, shape, reshape);
            break;
        }
        case ts::FLOAT64: {
            const double * psrc = stack.index(0)->sync(MemoryDevice(CPU)).data<double>();
            double * pdst = tensor->data<double>();
            transpose_run<double>(psrc, ncount, pdst, shape, reshape);
            break;
        }
        defalut: {
            throw Exception("transpose not support this data type");
            break;
        }
    }
    return 1;
}




}



using namespace ts;
TS_REGISTER_OPERATOR(Transpose, CPU, name::layer::transpose())
