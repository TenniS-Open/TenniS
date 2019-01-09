#ifndef TS_KERNELS_TRANSPOSE_H
#define TS_KERNELS_TRANSPOSE_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>
#include <cstring>
//#include <math.h>

#include <string.h>
#include <set>

namespace ts {


class Transpose : public ts::Operator {
public:

    using supper = ts::Operator;
    Transpose() {
        field("permute", OPTIONAL);
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

    template<typename T>
    void tranpose_run(T * psrc, int len, T* pdst,  const Shape &shape, const Shape &reshape, const Shape &permute) {

        Shape tmpshape;
        tmpshape.resize(shape.size());

        int index = 0;
        for(unsigned int i=0; i<len; i++) {
            Shape oldshape = to_coordinate(i, shape);
            for(int k=0; k<oldshape.size(); k++) {
                tmpshape[k] = oldshape[permute[k]];
            }

            index = to_index(reshape, tmpshape);

            if(index < 0) {
                throw ts::Exception("tranpose operator failed, index is invalid");
            }
            pdst[index] = psrc[i];
            //std::cout << "new index:" << index << ",index:" << i << ",type_len:" << type_len  << ",value:" << input_tensor->data<int>()[i] << std::endl;
            //tensor->data<int>()[index] = input_tensor->data<int>()[i];
            //std::cout << "new index:" << index << ",index:" << i << ",type_len:" << type_len << std::endl;
            //::memcpy(tensor->sync(memory_device()).data() + type_len * index, 
            //         input_tensor->sync(memory_device()).data() + type_len * i, type_len); 
        }
    } 

    virtual int run(ts::Stack &stack) {

        ts::Tensor::Prototype output_permute;
        ts::Tensor::Prototype output;
        //std::vector<ts::Tensor::Prototype> output;
        infer_private(stack, output_permute, output);

        ts::Tensor *input_tensor = stack.index(0);
        Shape shape = input_tensor->sizes();
        Shape permute = output_permute.sizes();
        Shape reshape = output.sizes();

        //int type_len = ts::type_bytes(input_tensor->dtype());

        stack.push(output, memory_device());

        ts::Tensor *tensor = stack.index(-1);
        ts::DTYPE type = stack.index(0)->dtype();

        unsigned int ncount = input_tensor->count();
        switch(type) {
            case ts::INT8: {
                char * psrc = stack.index(0)->sync(memory_device()).data<char>();
                char * pdst = tensor->sync(memory_device()).data<char>();
                tranpose_run<char>(psrc, ncount, pdst, shape, reshape, permute);
                break;
            }
            case ts::INT16: {
                short * psrc = stack.index(0)->sync(memory_device()).data<short>();
                short * pdst = tensor->sync(memory_device()).data<short>();
                tranpose_run<short>(psrc, ncount, pdst, shape, reshape, permute);
                break;
            }
            case ts::INT32: {
                int * psrc = stack.index(0)->sync(memory_device()).data<int>();
                int * pdst = tensor->sync(memory_device()).data<int>();
                tranpose_run<int>(psrc, ncount, pdst, shape, reshape, permute);
                break;
            }
            case ts::FLOAT32: {
                float * psrc = stack.index(0)->sync(memory_device()).data<float>();
                float * pdst = tensor->sync(memory_device()).data<float>();
                tranpose_run<float>(psrc, ncount, pdst, shape, reshape, permute);
                break;
            }
            case ts::FLOAT64: {
                double * psrc = stack.index(0)->sync(memory_device()).data<double>();
                double * pdst = tensor->sync(memory_device()).data<double>();
                tranpose_run<double>(psrc, ncount, pdst, shape, reshape, permute);
                break;
            }
            defalut: {
                throw ts::Exception("tranpose not support this data type");
                break;
            }
        }
 
        /* 
        //std::cout << "new type:" << type_str(tensor->dtype()) << std::endl;
        Shape tmpshape;
        tmpshape.resize(shape.size());

        int index = 0;
        unsigned int ncount = input_tensor->count();
        for(unsigned int i=0; i<ncount; i++) {
            Shape oldshape = to_coordinate(i, shape);
            for(int k=0; k<oldshape.size(); k++) {
                tmpshape[k] = oldshape[permute[k]];
            }

            index = to_index(reshape, tmpshape);

            if(index < 0) {

            }
            //std::cout << "new index:" << index << ",index:" << i << ",type_len:" << type_len  << ",value:" << input_tensor->data<int>()[i] << std::endl;
            //tensor->data<int>()[index] = input_tensor->data<int>()[i];
            //std::cout << "new index:" << index << ",index:" << i << ",type_len:" << type_len << std::endl;
            ::memcpy(tensor->sync(memory_device()).data() + type_len * index, 
                     input_tensor->sync(memory_device()).data() + type_len * i, type_len); 
        }
        */
        return 1;
    }

    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
        ts::Tensor::Prototype permute;
        output.resize(1);
        return infer_private(stack, permute, output[0]);
    }

private:
    int infer_private(ts::Stack &stack, ts::Tensor::Prototype &output_permute, ts::Tensor::Prototype &output) {
        int input_num = stack.size();
        if(input_num != 1) {
            throw ts::Exception("input parameters is more than one");
        }

        Shape shape = stack.index(0)->sizes();
        Shape permute;
        Shape reshape;

        if(shape.size()  < 1) {
            throw ts::Exception("input parameters dims is litter 1");
        }
 
        permute.resize(shape.size());
        reshape.resize(shape.size());

        if(!has("permute")){
            for(int j=0, i=shape.size() - 1; i>=0; i--,j++) {
                permute[j] = i;
                reshape[j] = shape[i];
                if(reshape[j] <= 0) {
                    throw ts::Exception("input tensor dim invalid"); 
                } 
            }
        }else {
            Tensor tensor_permute  = get("permute");
            if((tensor_permute.dims() != 1) && (tensor_permute.count() != shape.size())) {
                throw ts::Exception("permute parameter is not match input Tensor"); 
            } 

            if(tensor_permute.dtype() != ts::INT32) {
                throw ts::Exception("permute parameter only support INT32 dtype"); 
            }

            std::set<int> tmpset;
            for(int i=0; i<tensor_permute.count(); i++) {
                permute[i] = tensor_permute.sync(memory_device()).data<int>()[i];
                if(permute[i] < 0 || permute[i] >= shape.size()) {
                    throw ts::Exception("permute parameter is invalid"); 
                }
                tmpset.insert(permute[i]);
                reshape[i] = shape[permute[i]];

                if(reshape[i] <= 0) {
                    throw ts::Exception("input tensor dim invalid"); 
                } 
            }
            if(tmpset.size() != shape.size()) {
                throw::ts::Exception("permute parameter have duplicate value");
            }
        }

        //std::cout << "type:" << type_str(stack.index(0)->dtype()) << std::endl;
        //for(int i=0; i<permute.size(); i++)
        //    std::cout << "permute:" << permute[i] << ","  << std::endl;

        //for(int i=0; i<permute.size(); i++)
        //    std::cout << "new shape:" << reshape[i] << "," << std::endl;

        //output.resize(2);
        //output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), permute);
        //output[1] = ts::Tensor::Prototype(stack.index(0)->dtype(), reshape);

        output = ts::Tensor::Prototype(stack.index(0)->dtype(), reshape);
        output_permute = ts::Tensor::Prototype(stack.index(0)->dtype(), permute);
        return 1;
    }


};

}

#endif
