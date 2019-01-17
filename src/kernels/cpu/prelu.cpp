#include <kernels/cpu/prelu.h>
#include <core/tensor_builder.h>
#include "backend/name.h"

#include <algorithm>

namespace ts {

	Prelu::Prelu() : m_dim(-1) 
	{
		field(name::dim, OPTIONAL);
		field(name::slope, REQUIRED);
	}

	void Prelu::init()
	{
		supper::init();

		ts::Tensor slope = tensor::cast(FLOAT32,get(name::slope));

		if (slope.count() != 1)
		{
			TS_CHECK(has(name::dim)) << "Must input dim parameter" << ts::eject;
			m_dim = tensor::to_int(get(name::dim));

			float* slope_data = slope.data<float>();
			for (int i = 0; i < slope.count(); i++)
			{
				m_slope.emplace_back(*slope_data++);
			}
		}
		else
		{
			m_slope.emplace_back(tensor::to_float(slope));
		}
	}

	int Prelu::run(ts::Stack &stack)
	{
		std::vector<ts::Tensor::Prototype> output;

		this->infer(stack, output);
		stack.push(output[0], memory_device());

		auto dtype = stack.index(0)->dtype();
		int bytes = type_bytes(dtype);
		bool flag;
		switch (bytes)
		{
			case 1: flag = prelu<char>(stack); break;
			case 2: flag = prelu<short>(stack); break;
			case 4 : flag = prelu<float>(stack); break;
			case 8: flag = prelu<double>(stack); break;
			default:break;
		}
		//switch (dtype)
		//{
		//	case ts::FLOAT32:
		//	{
		//		flag = prelu<float>(stack);
		//		break;
		//	}
		//	case ts::FLOAT64:
		//	{
		//		flag = prelu<double>(stack);
		//		break;
		//	}
		//	default:break;
		//}
		return 1;
	}

	int Prelu::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output)
	{
		int input_num = stack.size();

		TS_AUTO_CHECK(input_num == 1);
		TS_AUTO_CHECK(stack.index(0)->dtype() == FLOAT32 || stack.index(0)->dtype() == FLOAT64);

		int slope_size = m_slope.size();

		TS_CHECK(slope_size == 1 || (m_dim >= 0 && m_dim < stack.index(0)->dims())) << "Prelu dim parameter should be greater or equale to 0 and less than input's dims" << ts::eject;

		TS_CHECK(slope_size == 1 || slope_size == stack.index(0)->sizes()[m_dim]) << "Parameter slope should be 1 or equal to the dim dimension of the input parameter" << ts::eject;
			
		output.resize(1);
		output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), stack.index(0)->sizes());

		return 1;
	}

	template<typename T>
	bool Prelu::prelu(ts::Stack &stack)
	{
		ts::Tensor& output_tensor = *stack.index(-1);
		auto output_shape = output_tensor.sizes();
		auto input_memory = stack.index(0)->sync(memory_device());
		auto device_type = input_memory.device();
		T* input_data = input_memory.data<T>();
		T* output_data = output_tensor.data<T>();

		int count = output_tensor.count();
		memcpy(output_data, device_type, count * sizeof(T), input_data, device_type, count * sizeof(T));

		if (m_slope.size() != 1)
		{
			int pre_dims = 1;
			for (int i = 0; i < m_dim; i++)
			{
				pre_dims *= output_shape[i];
			}
			int last_dims = 1;
			for (int i = m_dim + 1; i < output_shape.size(); i++)
			{
				last_dims *= output_shape[i];
			}
			int stride_dims = output_shape[m_dim] * last_dims;

			int pre_offset = 0;

			for (int i = 0; i < pre_dims; i++)
			{
				for (int j = 0; j < output_shape[m_dim]; j++)
				{
					pre_offset = i * output_shape[m_dim] * last_dims + j * last_dims;
					T val = static_cast<T>(m_slope[j]);
					for (int k = 0; k < last_dims; k++)
					{
						output_data[k + pre_offset] = std::max(output_data[k + pre_offset],T(0)) + val * std::min(output_data[k + pre_offset],T(0));
					}
				}
			}
		}
		else
		{
			float val = m_slope[0];
			for (int i = 0; i < output_tensor.count(); i++)
			{
				*output_data = std::max(*output_data, T(0)) + val * std::min(*output_data, T(0));
				output_data++;
			}
		}
		return true;
	}
	
}

using namespace ts;
TS_REGISTER_OPERATOR(Prelu, ts::CPU, name::layer::prelu())