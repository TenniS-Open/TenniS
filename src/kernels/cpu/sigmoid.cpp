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
		stack.push(output[0], memory_device());

		auto dtype = stack.index(0)->dtype();
		bool flag;
		switch (dtype)
		{
			case ts::FLOAT32:
			{
				flag = sigmoid<float>(stack);
				break;
			}
			case ts::FLOAT64:
			{
				flag = sigmoid<double>(stack);
				break;
			}
			default:break;
		}
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

		T* input_data = stack.index(0)->sync(memory_device()).data<T>();
		T* output_data = output_tensor.sync(memory_device()).data<T>();

		for (int i = 0; i < output_tensor.count(); i++)
		{
			*output_data = 1. / (1. + exp(-(*input_data)));
			input_data++;
			output_data++;
		}

		return true;
	}
}

using namespace ts;
TS_REGISTER_OPERATOR(Sigmoid, ts::CPU, name::layer::sigmoid())