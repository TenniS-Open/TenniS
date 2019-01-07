#ifndef TS_KERNELS_DIMSHUFFLE_H
#define TS_KERNELS_DIMSHUFFLE_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>
#include <cstring>
//#include <math.h>

#include <string.h>
#include <set>

namespace ts {


class Dimshuffle : public ts::Operator {
public:

    using supper = ts::Operator;
    Dimshuffle() {
        field("dim", REQUIRED);
        field("shuffle", REQUIRED);
    }

    virtual void init() {
        supper::init();
    }


    virtual int run(ts::Stack &stack) {
        ts::Tensor::Prototype output;
        int dim = 0;
        Shape shuffle;
        infer_private(stack, dim, shuffle, output);

        ts::Tensor *input_tensor = stack.index(0);
        Shape shape = input_tensor->sizes();
        Shape reshape = output.sizes();

        int type_len = ts::type_bytes(input_tensor->dtype());

        stack.push(output, memory_device());
        ts::Tensor *tensor = stack.index(-1);
      
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
        unsigned int newstride = backstride * shuffle.size();

        for(int k=0; k<preoffset; k++) {
            for(int i=0; i<shuffle.size(); i++) {
                ::memcpy(tensor->sync(memory_device()).data() + type_len * (k * newstride + i * backstride),
                         input_tensor->sync(memory_device()).data() + type_len * (k * stride + shuffle[i] * backstride), 
                         backstride * type_len); 
            }
        }

        return 1;
    }

    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
        Shape shuffle;
        int dim = 0;
        output.resize(1);
        return infer_private(stack, dim, shuffle, output[0]);
    }

private:
    int infer_private(ts::Stack &stack, int & dim, Shape & shuffle, ts::Tensor::Prototype &output) {
        int input_num = stack.size();
        if(input_num != 1) {
            throw ts::Exception("Dimshuffle input parameters is more than one");
        }

        Shape shape = stack.index(0)->sizes();

        if(shape.size()  < 1) {
            throw ts::Exception("Dimshuffle input parameters dims is litter 1");
        }
 
        if(!has("dim")){
            throw ts::Exception("Dimshuffle input parameter dim do not find"); 
        }

        if(!has("shuffle")){
            throw ts::Exception("Dimshuffle input parameter shuffle do not find"); 
        }
      
        Tensor tensor_dim = get("dim");
        if(tensor_dim.dims() != 1 || tensor_dim.count() != 1) {
            throw ts::Exception("Dimshuffle input parameter dim check failed"); 
        }

        if(tensor_dim.dtype() != ts::INT32) {
            throw ts::Exception("Dimshuffle input parameter dim only support INT32 type"); 
        }
        dim = tensor_dim.sync(memory_device()).data<int>()[0];
 
        Tensor tensor_shuffle = get("shuffle");
        if(tensor_shuffle.dims() != 1 || tensor_shuffle.count() < 1 || tensor_shuffle.dtype() != ts::INT32) {
            throw ts::Exception("Dimshuffle input parameter shuffle check failed"); 
        }

        if(dim < 0 || dim >= shape.size()) {
            throw ts::Exception("Dimshuffle input parameter dim value check failed"); 
        }

        if(tensor_shuffle.dtype() != ts::INT32) {
            throw ts::Exception("Dimshuffle input parameter shuffle only support INT32 type"); 
        }
        shuffle.resize(tensor_shuffle.count());
        for(int i=0; i<tensor_shuffle.count(); i++) {
            shuffle[i] = tensor_shuffle.sync(memory_device()).data<int>()[i];
            if(shuffle[i] < 0 || shuffle[i] >= shape[dim]) {
                throw ts::Exception("Dimshuffle shuffle parameter is invalid"); 
            }
        }

        if(shuffle.size() != shape[dim]) {
            shape[dim] = shuffle.size();
        }
        output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
        return 1;
    }


};

}

#endif
