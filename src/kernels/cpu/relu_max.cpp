#include <kernels/cpu/relu_max.h>
#include <core/tensor_builder.h>
#include <algorithm>

namespace ts {

	void ReluMax::init()
	{
		supper::init();

		if (has("max"))
		{
			m_max = tensor::to_float(get("max"));
		}
		else
			throw ts::Exception("Missing parameter max");
	}

	int ReluMax::run(ts::Stack &stack)
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
				flag = relu_max<float>(stack);
				break;
			}
			case ts::FLOAT64:
			{
				flag = relu_max<double>(stack);
				break;
			}
			default:
			{
				throw ts::Exception("relu_max only support FLOAT32 and FLOAT64 type");
				break;
			}
		}
		if (!flag)
			throw ts::Exception("relu_max failed!");
		return 1;
	}

	int ReluMax::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output)
	{
		int input_num = stack.size();

		if (input_num != 1)
			throw ts::Exception("Input parameter should be one!");

		if (stack.index(0)->dtype() != FLOAT32 && stack.index(0)->dtype() != FLOAT64)
			throw ts::Exception("Input parameter should be float or double");

		output.resize(1);
		output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), stack.index(0)->sizes());

		return 1;
	}

	template<typename T>
	bool ReluMax::relu_max(ts::Stack &stack)
	{
		ts::Tensor& output_tensor = *stack.index(-1);

		T* input_data = stack.index(0)->sync(memory_device()).data<T>();
		T* output_data = output_tensor.sync(memory_device()).data<T>();
		if (output_data == nullptr)
			return false;

		for (int i = 0; i < output_tensor.count(); i++)
		{
			*output_data = std::min(*input_data > 0 ? *input_data : 0 , m_max);
			input_data++;
			output_data++;
		}

		return true;
	}

}