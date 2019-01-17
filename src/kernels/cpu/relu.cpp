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
		switch (dtype)
		{
			case ts::FLOAT32:
			{
				flag = relu<float>(stack);
				break;
			}
			case ts::FLOAT64:
			{
				flag = relu<double>(stack);
				break;
			}
			default:break;
		}

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
		T* input_data = stack.index(0)->sync(memory_device()).data<T>();
		T* output_data = output_tensor.sync(memory_device()).data<T>();
		//::memcpy(output_data, input_data, stack.index(0)*sizeof(T));
		int count = output_tensor.count();
		memcpy(output_data, MemoryDevice(CPU), count * sizeof(T), input_data, MemoryDevice(CPU), count * sizeof(T));

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
