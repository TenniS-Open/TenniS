#include <kernels/cpu/fused_batch_norm.h>

#include <kernels/cpu/batch_norm.h>
#include <kernels/cpu/batch_scale.h>
#include <core/tensor_builder.h>

#include <global/operator_factory.h>
#include <backend/name.h>


namespace ts {



//////////////////////////////////////////////
Fused_Batch_Norm::Fused_Batch_Norm() {
    field("epsilon", OPTIONAL);
    field("dim", REQUIRED);
} 

void Fused_Batch_Norm::init() {
    supper::init();

    if(has("dim")){
        const Tensor& tensor_dim = get("dim");
        m_dim = ts::tensor::to_int(tensor_dim);
    }else {
        throw ts::Exception("fused_batch_norm missed dim parameter ");
    }
  
}   

void Fused_Batch_Norm::infer_private(ts::Stack &stack, ts::Tensor::Prototype &output) {
    int input_num = stack.size();
    if(input_num != 5) {
        throw ts::Exception("fused_batch_norm must have 5 input parameters");
    }

    const Shape& shape = stack.index(0)->sizes();

    if(m_dim < 0 || m_dim >= shape.size() ) {
        throw ts::Exception("fused_batch_norm dim parameter check failed");
    }
   
    const Shape& gamma_shape = stack.index(1)->sizes();
    if(gamma_shape.size()  != 1 ) {
        throw ts::Exception("fused_batch_norm gamma parameter's dims is not 1");
    }

    const Shape& beta_shape = stack.index(2)->sizes();
    std::cout << "beta:" << beta_shape.size() << std::endl;
    if(beta_shape.size()  != 1 ) {
        throw ts::Exception("fused_batch_norm beta parameter's dims is not 1");
    }
    const Shape& mean_shape = stack.index(3)->sizes();
    if(mean_shape.size()  != 1 ) {
        throw ts::Exception("fused_batch_norm mean parameter's dims is not 1");
    }

    const Shape& variance_shape = stack.index(4)->sizes();
    if(variance_shape.size()  != 1 ) {
        throw ts::Exception("fused_batch_norm variance parameter's dims is not 1");
    }

    if(gamma_shape[0] != shape[m_dim] || beta_shape[0] != shape[m_dim] ||
       variance_shape[0] != shape[m_dim] || mean_shape[0] != shape[m_dim]) {
        throw ts::Exception("fused_batch_norm gamma,beta, mean and variance parameters check failed");
    }

    output = ts::Tensor::Prototype(stack.index(0)->dtype(), shape);
    return;
}



int Fused_Batch_Norm::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
    output.resize(1);
    infer_private(stack, output[0]);
    return 1;
}


int Fused_Batch_Norm::run(ts::Stack &stack) {
    std::vector<ts::Tensor::Prototype> output;
    output.resize(1);
    infer_private(stack, output[0]);

    ts::Tensor *input_tensor =  stack.index(0);
    ts::Tensor *gamma_tensor  = stack.index(1);
    ts::Tensor *beta_tensor  = stack.index(2);

    ts::Tensor *mean_tensor  = stack.index(3);
    ts::Tensor *variance_tensor  = stack.index(4);
    //stack.push(output[0], memory_device());
    //ts::Tensor *tensor = stack.index(-1);


    auto bn = ts::OperatorCreator::Query(computing_device().type(), "batch_norm");
    auto bc = ts::OperatorCreator::Query(computing_device().type(), "batch_scale");
 
   if(bn == nullptr) {
        bn = ts::OperatorCreator::Query(memory_device().type(), "batch_norm");
        if(bn == nullptr) {
            bn = ts::OperatorCreator::Query(ts::CPU, "batch_norm");
        }
    }

    if(bc == nullptr) {
        bc = ts::OperatorCreator::Query(memory_device().type(), "batch_scale");
        if(bc == nullptr) {
            bc = ts::OperatorCreator::Query(ts::CPU, "batch_scale");
        }
    }

    if(bn == nullptr || bc == nullptr) {
        throw ts::Exception("fused_batch_norm do not find batch_norm or batch_scale operator"); 
    }


    auto batch_norm = bn();
    auto batch_scale = bc();

    std::cout << "mean_tensor:" << mean_tensor->data<float>()[0] << "," << mean_tensor->data<float>()[1] << "," << mean_tensor->data<float>()[2] << std::endl;
    std::cout << "variance_tensor:" << variance_tensor->data<float>()[0] << std::endl;
    std::cout << "variance_tensor:" << variance_tensor->data<float>()[1] << std::endl;
    std::cout << "variance_tensor:" << variance_tensor->data<float>()[2] << std::endl;
   
    if(has("epsilon")){
        batch_norm->set("epsilon",get("epsilon"));
    }
 
    batch_norm->set("dim", get("dim"));
    stack.push(*input_tensor);
    stack.push(*mean_tensor);
    stack.push(*variance_tensor);

    batch_norm->init();

    //stack.push_base(-3);
    //batch_norm->run(stack);
    //stack.pop_base(); 
    RunOperator(batch_norm, stack, 3);

    batch_scale->set("dim", get("dim"));
    stack.push(*gamma_tensor);
    stack.push(*beta_tensor);

    batch_scale->init();

    RunOperator(batch_scale, stack, 3);
    //stack.push_base(-3);
    //batch_scale->run(stack);
    //stack.pop_base(); 

    return 1;
}





/////////////////////////////////////////////////
TS_REGISTER_OPERATOR(Fused_Batch_Norm, ts::CPU, ts::name::layer::fused_batch_norm())


}

