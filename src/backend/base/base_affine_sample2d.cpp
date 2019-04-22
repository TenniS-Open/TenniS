//
// Created by kier on 2019/2/15.
//

#include "backend/base/base_affine_sample2d.h"

#include <utils/assert.h>
#include <numeric>

#include <backend/name.h>
#include <core/tensor_builder.h>


namespace ts {
    namespace base {



        Affine_Sample2D::Affine_Sample2D() {
            field(name::type, REQUIRED);
            field(name::dim, REQUIRED);
        }

        void Affine_Sample2D::init() {
            supper::init();

            Tensor type_tensor = tensor::cast(INT32, get(name::type));
            Tensor dim_tensor = tensor::cast(INT32, get(name::dim));

            TS_AUTO_CHECK(type_tensor.has_shape({1}));
            TS_AUTO_CHECK(dim_tensor.has_shape({1}));

            m_type = Affine_Sample2DType(tensor::to_int(type_tensor));
            m_dim = tensor::to_int(dim_tensor);

            TS_AUTO_CHECK((m_type >= Affine_Sample2DType::LINEAR) && (m_type <= Affine_Sample2DType::NEAREST));

        }

        int Affine_Sample2D::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 3);
            auto &x = stack[0];

            Shape input_shape = x.sizes();

            TS_AUTO_CHECK(input_shape.size() >= 2);
            int dim = m_dim;
            if(dim < 0) {
               dim = input_shape.size() + dim;
            }

            std::cout << dim << std::endl;
            TS_AUTO_CHECK((dim >= 0) && (dim + 1  < input_shape.size()));

            auto size_tensor = tensor::cast(INT32, stack[1]);

            TS_AUTO_CHECK(size_tensor.count() == 2);
           
            const int32_t * pdata = size_tensor.data<int32_t>(); 

            input_shape[dim] = pdata[0];
            input_shape[dim +1] = pdata[1];

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), input_shape);

            return 1;
        }

        int Affine_Sample2D::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 3);

            std::vector<Tensor::Prototype> output;
            infer(stack, output);
            

            auto memory_device = running_memory_device();
            auto x = stack[0].view(memory_device);

            Shape input_shape = x.sizes();
            TS_AUTO_CHECK(input_shape.size() >= 2);
            int dim = m_dim;
            if(dim < 0) {
               dim = input_shape.size() + dim;
            }
            TS_AUTO_CHECK((dim >= 0) && (dim + 1  < input_shape.size()));



            auto out = *stack.push(output[0], memory_device);
            
            
            // auto & size_tensor = stack[1];

            auto affine_tensor = tensor::cast(FLOAT32, stack[2]);

            Shape affine_shape = affine_tensor.sizes();
            
            TS_AUTO_CHECK((affine_shape.size() == 2) && (affine_shape[0] == 3) && (affine_shape[1] == 3));

            // int * pdata = size_tensor.data<int32_t>();
            float * paffine = affine_tensor.data<float>(); 
            
            affine_sample_run((const Tensor &)x, paffine[0], paffine[1], paffine[2], paffine[3], paffine[4], paffine[5], paffine[6], paffine[7], paffine[8],
                              m_type, dim, out);
            return 1;
        }
    }
}
