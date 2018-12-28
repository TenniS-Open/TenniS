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


    static std::vector<int> to_coordinate(int index, std::vector<int> &shape)
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
 

    virtual int run(ts::Stack &stack) {

        std::vector<ts::Tensor::Prototype> output;
        infer(stack, output);

        ts::Tensor *input_tensor = stack.index(0);
        Shape shape = input_tensor->sizes();
        Shape permute = output[0].sizes();
        Shape reshape = output[1].sizes();

        int type_len = ts::type_bytes(input_tensor->dtype());

        stack.push(output[1]);

        ts::Tensor *tensor = stack.index(-1);
      
        //std::cout << "new type:" << type_str(tensor->dtype()) << std::endl;
        Shape tmpshape;
        tmpshape.resize(shape.size());

        unsigned int index = 0;
        unsigned int ncount = input_tensor->count();
        for(unsigned int i=0; i<ncount; i++) {
            Shape oldshape = to_coordinate(i, shape);
            for(int k=0; k<oldshape.size(); k++) {
                tmpshape[k] = oldshape[permute[k]];
            }

            index = to_index(reshape, tmpshape);

            //std::cout << "new index:" << index << ",index:" << i << ",type_len:" << type_len  << ",value:" << input_tensor->data<int>()[i] << std::endl;
            //tensor->data<int>()[index] = input_tensor->data<int>()[i];
            //std::cout << "new index:" << index << ",index:" << i << ",type_len:" << type_len << std::endl;
            ::memcpy(tensor->data() + type_len * index, input_tensor->data() + type_len * i, type_len); 
        }
        return 1;
    }

    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
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
                permute[i] = tensor_permute.data<int>()[i];
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

        std::cout << "type:" << type_str(stack.index(0)->dtype()) << std::endl;
        for(int i=0; i<permute.size(); i++)
            std::cout << "permute:" << permute[i] << ","  << std::endl;

        for(int i=0; i<permute.size(); i++)
            std::cout << "new shape:" << reshape[i] << "," << std::endl;

        output.resize(2);
        output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), permute);
        output[1] = ts::Tensor::Prototype(stack.index(0)->dtype(), reshape);
        return 1;
    }



};

}

#endif
