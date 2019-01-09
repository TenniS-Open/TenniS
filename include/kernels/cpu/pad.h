#ifndef TS_KERNELS_PAD_H
#define TS_KERNELS_PAD_H

#include <global/operator_factory.h>
#include <core/tensor.h>

#include <core/dtype.h>
#include <runtime/stack.h>
#include <cstring>
//#include <math.h>

#include <string.h>
#include <set>

namespace ts {


class Pad : public ts::Operator {
public:

    using supper = ts::Operator;
    Pad() {
        field("padding_value", OPTIONAL);
    }

    virtual void init() {
        supper::init();
    }


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


    template <typename T>
    void padding_run(T * psrc, int len, T* pdst, int* padding, const Shape &shape, const Shape &reshape, int padding_value) {
        int index = 0;
        Shape tmpshape;
        tmpshape.resize(shape.size());
        for(unsigned int i=0; i<len; i++) {
            Shape oldshape = to_coordinate(i, reshape);

            for(int k=0; k<oldshape.size(); k++) {
                tmpshape[k] = oldshape[k] - padding[2 * k];// - padding[2 * k +1];       
            }
            
            index = to_index(shape, tmpshape);
            //std::cout << "i;" << i << ",index:" << index;
            //std::cout << "," << oldshape[0] << "," << oldshape[1] << "," << oldshape[2] << "," << oldshape[3] << std::endl;
            if(index >= 0 ) {
                pdst[i] = psrc[index];
            }else  {
                pdst[i] = padding_value;
            }
        }
    } 

    virtual int run(ts::Stack &stack) {
        ts::Tensor::Prototype output;
        int padding_value = 0;
        infer_private(stack, output, padding_value);

        ts::Tensor *input_tensor = stack.index(0);
        Shape shape = input_tensor->sizes();
        Shape reshape = output.sizes();

        int type_len = ts::type_bytes(input_tensor->dtype());

        stack.push(output, memory_device());
        ts::Tensor *tensor = stack.index(-1);

        int * padding = stack.index(1)->sync(memory_device()).data<int>();
 
        Shape tmpshape;
        tmpshape.resize(shape.size());

        ts::DTYPE type = stack.index(0)->dtype();

        unsigned int ncount = tensor->count();
        switch(type) {
            case ts::INT8: {
                char * psrc = stack.index(0)->sync(memory_device()).data<char>();
                char * pdst = tensor->sync(memory_device()).data<char>();
                padding_run<char>(psrc, ncount, pdst, padding, shape, reshape, padding_value);
                break;
            }
            case ts::INT16: {
                short * psrc = stack.index(0)->sync(memory_device()).data<short>();
                short * pdst = tensor->sync(memory_device()).data<short>();
                padding_run<short>(psrc, ncount, pdst, padding, shape, reshape, padding_value);
                break;
            }
            case ts::INT32: {
                int * psrc = stack.index(0)->sync(memory_device()).data<int>();
                int * pdst = tensor->sync(memory_device()).data<int>();
                padding_run<int>(psrc, ncount, pdst, padding, shape, reshape, padding_value);
                break;
            }
            case ts::FLOAT32: {
                float * psrc = stack.index(0)->sync(memory_device()).data<float>();
                float * pdst = tensor->sync(memory_device()).data<float>();
                padding_run<float>(psrc, ncount, pdst, padding, shape, reshape, padding_value);
                break;
            }
            case ts::FLOAT64: {
                double * psrc = stack.index(0)->sync(memory_device()).data<double>();
                double * pdst = tensor->sync(memory_device()).data<double>();
                padding_run<double>(psrc, ncount, pdst, padding, shape, reshape, padding_value);
                break;
            }
            defalut: {
                throw ts::Exception("pad not support this data type");
                break;
            }
        }
 
        return 1;
    }

    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
        //ts::Tensor::Prototype permute;
        output.resize(1);
        int padding_value = 0;
        return infer_private(stack, output[0], padding_value);
    }

private:
    int infer_private(ts::Stack &stack, ts::Tensor::Prototype &output, int &padding_value) {
        int input_num = stack.size();
        if(input_num != 2) {
            throw ts::Exception("pad only support two input parameters");
        }

        Shape shape = stack.index(0)->sizes();
        Shape pad_shape = stack.index(1)->sizes();;
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
 
        padding_value = 0;
        if(has("padding_value")){
            Tensor tensor_padding_value = get("padding_value");
            if(tensor_padding_value.dims() != 1 || tensor_padding_value.dtype() != ts::INT32 || tensor_padding_value.count() != 1) {
                throw ts::Exception("pad input parameter padding_value check failed");
            }
            padding_value = tensor_padding_value.sync(memory_device()).data<int>()[0];
        }


        reshape.resize(shape.size());

        int * padding = stack.index(1)->sync(memory_device()).data<int>();
        std::cout << "reshape:";
        for(int i=0; i<shape.size(); i++) {
           reshape[i] = shape[i] + padding[2 * i] + padding[2 * i + 1];
           if(reshape[i] <= 0) {
               throw ts::Exception("pad padding parameter value check failed");
           }

           std::cout << reshape[i] << ",";
        }
        std::cout << std::endl;

        output = ts::Tensor::Prototype(stack.index(0)->dtype(), reshape);
        return 1;
    }


};

}

#endif
