#ifndef TS_KERNELS_DIMSHUFFLE_H
#define TS_KERNELS_DIMSHUFFLE_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>
#include <cstring>
#include <math.h>



namespace ts {


class Reshape : public ts::Operator {
public:

    using supper = ts::Operator;
    Reshape() {
        field("shape", REQUIRED);
    }

    virtual void init() {
        supper::init();
    }

    virtual int run(ts::Stack &stack) {
        std::vector<ts::Tensor::Prototype> output;
        infer(stack, output);
        if(output.size() < 1) {
            throw ts::Exception("Reshape infer() failed");
        }
        //ts::Tensor::Prototype prototype(stack.index(0)->dtype(),resshape);
        //output[0] = prototype;
        //stack.push(output[0]);

        //auto &tensor = *stack.index(-1);
        Tensor tensor = stack.index(0)->reshape(output[0].sizes());
        stack.push(tensor);
        return 1;
    }

    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
        if(!has("shape")){
            throw ts::Exception("shape parameter do not find");
        }

        Tensor tensor_shape  = get("shape");

        if(tensor_shape.dtype() != INT32){
            throw ts::Exception("shape parameter dtype is not INT32");
        }

        int input_num = stack.size();
        if(input_num != 1) {
            throw ts::Exception("input parameters is more than one");
        }

        Shape shape = stack.index(0)->sizes();

        int dims = stack.index(0)->dims();
        int ncount = tensor_shape.count();
        bool bfind = false;

        if(ncount < 1) {
            throw ts::Exception("shape parameter dims must bigger than 0"); 
        }

        int nindex = -1;
        Shape reshape(ncount);
        unsigned int ntotal = 1;
        for(int i=0; i<ncount; i++) {
            reshape[i] = tensor_shape.data<int>()[i];
            std::cout << "i:" << reshape[i] << std::endl;
            if(reshape[i] > 0) {
                    ntotal *= reshape[i];
            }
            if(tensor_shape.data<int>()[i] <= 0) {
                if(bfind) {
                    throw ts::Exception("shape parameters only one less than 0 ");
                }else {
                    bfind = true;
                    nindex = i;
                }                       
            }
        }

        if(ntotal <= 0) {
            throw ts::Exception("shape parameter dims is invalid"); 
        }
                
        unsigned int nsize = stack.index(0)->count();
         
        if(nindex >= 0) {
            reshape[nindex] = nsize / ntotal;
            ntotal *= reshape[nindex];
        }

        if(nsize != ntotal) {
            throw ts::Exception("shape parameter is invalid");
        }

        output.resize(1);
        output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), reshape);
        return 1;
    }



};

}

#endif
