#include <kernels/cpu/relu.h>
#include <algorithm>

#include "backend/name.h"

namespace ts {

	void Relu::init()
	{
		supper::init();
	}

	int Relu::run(ts::Stack &stack)
	{
		std::vector<ts::Tensor::Prototype> output;

		this->infer(stack, output);
		stack.push(output[0], memory_device());

		auto dtype = stack.index(0)->dtype();
		bool flag;
		int bytes = type_bytes(dtype);
		switch (bytes)
		{
		case 1: flag = relu<char>(stack); break;
		case 2: flag = relu<short>(stack); break;
		case 4: flag = relu<float>(stack); break;
		case 8: flag = relu<double>(stack); break;
		default:break;
		}
		//switch (dtype)
		//{
		//	case ts::FLOAT32:
		//	{
		//		flag = relu<float>(stack);
		//		break;
		//	}
		//	case ts::FLOAT64:
		//	{
		//		flag = relu<double>(stack);
		//		break;
		//	}
		//	default:break;
		//}

		return 1;
	}

	int Relu::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output)
	{
		int input_num = stack.size();
		TS_AUTO_CHECK(input_num == 1);

		TS_AUTO_CHECK(stack.index(0)->dtype() == FLOAT32 || stack.index(0)->dtype() == FLOAT64);

		output.resize(1);
		output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), stack.index(0)->sizes());

		return 1;
	}

	template<typename T>
	bool Relu::relu(ts::Stack &stack)
	{
		ts::Tensor& output_tensor = *stack.index(-1);
		auto output_shape = output_tensor.sizes();
		auto input_memory = stack.index(0)->sync(memory_device());
		auto device_type = input_memory.device();
		T* input_data = input_memory.data<T>();
		T* output_data = output_tensor.data<T>();
		int count = output_tensor.count();
		memcpy(output_data, device_type, count * sizeof(T), input_data, device_type, count * sizeof(T));

		for (int i = 0; i < output_tensor.count(); i++)
		{
			T val = *output_data;
			*output_data = std::max(val,T(0.0));
			output_data++;
		}

		return true;
	}
}

using namespace ts;
TS_REGISTER_OPERATOR(Relu, ts::CPU, name::layer::relu())
