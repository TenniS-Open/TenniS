#include <kernels/cpu/softmax.h>
#include <core/tensor_builder.h>
#include "backend/name.h"
#include <algorithm>
#include <math.h>

namespace ts {

	Softmax::Softmax() :m_dim(-1)
	{
		field(name::dim, REQUIRED);
		Tensor default_smooth = tensor::from<bool>(true);
		field(name::smooth, OPTIONAL, default_smooth);
	}

	void Softmax::init()
	{
		supper::init();

		auto dim_tensor = get(name::dim);
		m_dim = tensor::to_int(dim_tensor);

		m_smooth = tensor::to_int(get(name::smooth)) ? true:false;

		TS_AUTO_CHECK(m_dim >= 0);
	}

	int Softmax::run(ts::Stack &stack)
	{
		std::vector<ts::Tensor::Prototype> output;

		this->infer(stack, output);
		stack.push(output[0], MemoryDevice(CPU));

		auto dtype = stack.index(0)->dtype();
		bool flag;
		int bytes = type_bytes(dtype);
		switch (bytes)
		{
		case 1: flag = softmax<char>(stack); break;
		case 2: flag = softmax<short>(stack); break;
		case 4: flag = softmax<float>(stack); break;
		case 8: flag = softmax<double>(stack); break;
		default:break;
		}
		//switch (dtype)
		//{
		//	case ts::FLOAT32:
		//	{
		//		flag = softmax<float>(stack);
		//		break;
		//	}
		//	case ts::FLOAT64:
		//	{
		//		flag = softmax<double>(stack);
		//		break;
		//	}
		//	default:break;
		//}
		return 1;
	}

	int Softmax::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output)
	{
		int input_num = stack.size();

		TS_AUTO_CHECK(input_num == 1);
		TS_AUTO_CHECK(stack.index(0)->dtype() == FLOAT32 || stack.index(0)->dtype() == FLOAT64);

		TS_AUTO_CHECK(m_dim >=0 && m_dim < stack.index(0)->dims());

		output.resize(1);
		output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), stack.index(0)->sizes());

		return 1;
	}

	template<typename T>
	bool Softmax::softmax(ts::Stack &stack)
	{
		ts::Tensor& output_tensor = *stack.index(-1);
		auto output_shape = output_tensor.sizes();

		int pre_num = 1;
		for (int i = 0; i < m_dim; i++)
		{
			pre_num *= output_shape[i];
		}
		int inner_num = 1;
		for (int i = m_dim+1; i < output_shape.size(); i++)
		{
			inner_num *= output_shape[i];
		}

		int axis = output_shape[m_dim];

		auto input_memory = stack.index(0)->sync(MemoryDevice(CPU));
		auto device_type = input_memory.device();
		T* input_data = input_memory.data<T>();
		T* output_data = output_tensor.data<T>();

		int count = output_tensor.count();
		memcpy(output_data, device_type, count * sizeof(T), input_data, device_type, count * sizeof(T));

		int scale_data_size = output_tensor.count() / axis;
		int denominator_data_size = scale_data_size;

		Shape scale_shape;
		scale_shape.resize(1);
		scale_shape[0] = scale_data_size;
		Tensor scale_tensor(MemoryDevice(CPU), output_tensor.dtype(), scale_shape);
		T* scale_data = scale_tensor.data<T>();
			 
		Shape denominator_shape;
		denominator_shape.resize(1);
		denominator_shape[0] = denominator_data_size;
		Tensor denominator_tensor(MemoryDevice(CPU), output_tensor.dtype(), denominator_shape);
		T* denominator_data = denominator_tensor.data<T>();

		for (int i = 0; i < pre_num; i++)
		{
			std::memset(denominator_data, 0, scale_data_size * sizeof(T));
			if (m_smooth)
			{
				//Caculate max value
				memcpy(scale_data, device_type, inner_num * sizeof(T), input_data + i * axis * inner_num, device_type, inner_num * sizeof(T));
				//std::memcpy(scale_data,input_data + i * axis * inner_num, inner_num * sizeof(T));
				for (int j = 0; j < axis; j++)
				{
					for (int k = 0; k < inner_num; k++)
					{
						scale_data[k] = std::max(scale_data[k], input_data[i*axis*inner_num + j*inner_num + k]);
					}
				}
				//Caculate numerator and denominator
				for (int j = 0; j < axis; j++)
				{
					for (int k = 0; k < inner_num; k++)
					{
						output_data[i*axis*inner_num + j*inner_num + k] = output_data[i*axis*inner_num + j*inner_num + k] - scale_data[k];
						output_data[i*axis*inner_num + j*inner_num + k] = exp(output_data[i*axis*inner_num + j*inner_num + k]);
						denominator_data[k] += output_data[i*axis*inner_num + j*inner_num + k];
					}
				}
			}
			else
			{
				//Caculate numerator and denominator
				for (int j = 0; j < axis; j++)
				{
					for (int k = 0; k < inner_num; k++)
					{
						output_data[i*axis*inner_num + j*inner_num + k] = output_data[i*axis*inner_num + j*inner_num + k];
						output_data[i*axis*inner_num + j*inner_num + k] = exp(output_data[i*axis*inner_num + j*inner_num + k]);
						denominator_data[k] += output_data[i*axis*inner_num + j*inner_num + k];
					}
				}
			}
			//Caculte output
			for (int j = 0; j < axis; j++)
			{
				for (int k = 0; k < inner_num; k++)
				{
					output_data[i*axis*inner_num + j*inner_num + k] = output_data[i*axis*inner_num + j*inner_num + k] / denominator_data[k];
				}
			}
			
		}
		
		return true;
	}
}

using namespace ts;
TS_REGISTER_OPERATOR(Softmax, ts::CPU, name::layer::softmax())