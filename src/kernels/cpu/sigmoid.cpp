#include <kernels/cpu/sigmoid.h>
#include <math.h>

#include "backend/name.h"

namespace ts {

	void Sigmoid::init()
	{
		supper::init();
	}

	int Sigmoid::run(ts::Stack &stack)
	{
		std::vector<ts::Tensor::Prototype> output;

		this->infer(stack, output);
		stack.push(output[0], MemoryDevice(CPU));

		auto dtype = stack.index(0)->dtype();
		bool flag;
		int bytes = type_bytes(dtype);
		switch (bytes)
		{
			case 1: flag = sigmoid<char>(stack); break;
			case 2: flag = sigmoid<short>(stack); break;
			case 4: flag = sigmoid<float>(stack); break;
			case 8: flag = sigmoid<double>(stack); break;
			default:break;
		}
		//switch (dtype)
		//{
		//	case ts::FLOAT32:
		//	{
		//		flag = sigmoid<float>(stack);
		//		break;
		//	}
		//	case ts::FLOAT64:
		//	{
		//		flag = sigmoid<double>(stack);
		//		break;
		//	}
		//	default:break;
		//}
		return 1;
	}

	int Sigmoid::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output)
	{
		int input_num = stack.size();

		TS_AUTO_CHECK(input_num == 1);
		TS_AUTO_CHECK(stack.index(0)->dtype() == FLOAT32 || stack.index(0)->dtype() == FLOAT64);

		output.resize(1);
		output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), stack.index(0)->sizes());

		return 1;
	}

	template<typename T>
	bool Sigmoid::sigmoid(ts::Stack &stack)
	{
		ts::Tensor& output_tensor = *stack.index(-1);

		auto input_memory = stack.index(0)->sync(MemoryDevice(CPU));
		auto device_type = input_memory.device();
		T* input_data = input_memory.data<T>();
		T* output_data = output_tensor.data<T>();

		int count = output_tensor.count();
		memcpy(output_data, device_type, count * sizeof(T), input_data, device_type, count * sizeof(T));

		for (int i = 0; i < output_tensor.count(); i++)
		{
			T val = *output_data;
			*output_data = 1. / (1. + exp(-(val)));
			output_data++;
		}

		return true;
	}
}

using namespace ts;
TS_REGISTER_OPERATOR(Sigmoid, ts::CPU, name::layer::sigmoid())