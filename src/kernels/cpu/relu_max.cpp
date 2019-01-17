#include <kernels/cpu/relu_max.h>
#include <core/tensor_builder.h>
#include "backend/name.h"
#include <algorithm>

namespace ts {

	ReluMax::ReluMax()
	{
		field(name::max, REQUIRED);
	}

	void ReluMax::init()
	{
		supper::init();
		m_max = tensor::to_float(get(name::max));
	}

	int ReluMax::run(ts::Stack &stack)
	{
		std::vector<ts::Tensor::Prototype> output;

		this->infer(stack, output);
		stack.push(output[0], memory_device());

		auto dtype = stack.index(0)->dtype();
		bool flag;
		int bytes = type_bytes(dtype);
		switch (bytes)
		{
			case 1: flag = relu_max<char>(stack); break;
			case 2: flag = relu_max<short>(stack); break;
			case 4: flag = relu_max<float>(stack); break;
			case 8: flag = relu_max<double>(stack); break;
			default:break;
		}
		//switch (dtype)
		//{
		//	case ts::FLOAT32:
		//	{
		//		flag = relu_max<float>(stack);
		//		break;
		//	}
		//	case ts::FLOAT64:
		//	{
		//		flag = relu_max<double>(stack);
		//		break;
		//	}
		//	default:break;
		//}

		return 1;
	}

	int ReluMax::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output)
	{
		int input_num = stack.size();

		TS_AUTO_CHECK(input_num == 1);
		TS_AUTO_CHECK(stack.index(0)->dtype() == FLOAT32 || stack.index(0)->dtype() == FLOAT64);

		output.resize(1);
		output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), stack.index(0)->sizes());

		return 1;
	}

	template<typename T>
	bool ReluMax::relu_max(ts::Stack &stack)
	{
		ts::Tensor& output_tensor = *stack.index(-1);

		auto input_memory = stack.index(0)->sync(memory_device());
		auto device_type = input_memory.device();
		T* input_data = input_memory.data<T>();
		T* output_data = output_tensor.data<T>();

		int count = output_tensor.count();
		memcpy(output_data, device_type, count * sizeof(T), input_data, device_type, count * sizeof(T));

		for (int i = 0; i < output_tensor.count(); i++)
		{
			//*output_data = std::min(*input_data > 0 ? *input_data : 0 , m_max);
			T val = *output_data;
			*output_data = std::min(std::max(val,T(0.0)), (T)m_max);
			output_data++;
		}

		return true;
	}
}

using namespace ts;
TS_REGISTER_OPERATOR(ReluMax, ts::CPU, name::layer::relu_max())