#include <kernels/cpu/prelu.h>
#include <core/tensor_builder.h>
#include <algorithm>

namespace ts {

	void Prelu::init()
	{
		supper::init();
		if (!has("slope"))
			throw ts::Exception("Must input slope parameter");

		ts::Tensor slope = get("slope");

		if (slope.count() != 1)
		{
			if (!has("dim"))
				throw ts::Exception("Must input dim parameter");
			m_dim = tensor::to_int(get("dim"));
			auto dtype = slope.dtype();
			if (dtype != FLOAT32)
				throw ts::Exception("Parameter slope should be float");

			auto slope_mem = slope.sync(memory_device());
			for (int i = 0; i < slope.count(); i++)
			{
				m_slope.emplace_back(static_cast<float>(slope_mem.data<float>()[i]));
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
		bool flag;
		switch (dtype)
		{
			case ts::FLOAT32:
			{
				flag = prelu<float>(stack);
				break;
			}
			case ts::FLOAT64:
			{
				flag = prelu<double>(stack);
				break;
			}
			default:
			{
				throw ts::Exception("prelu only support FLOAT32 and FLOAT64 type");
				break;
			}
		}
		if (!flag)
			throw ts::Exception("prelu failed!");
		return 1;
	}

	int Prelu::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output)
	{
		int input_num = stack.size();

		if (input_num != 1)
			throw ts::Exception("Input parameter should be one!");

		if (stack.index(0)->dtype() != FLOAT32 && stack.index(0)->dtype() != FLOAT64)
			throw ts::Exception("Input parameter should be float or double");

		int slope_size = m_slope.size();

		if (slope_size != 1 && m_dim < 0 || m_dim >= stack.index(0)->dims()) {
			throw ts::Exception("Prelu dim parameter check failed");
		}

		if (slope_size != 1 && slope_size != stack.index(0)->sizes()[m_dim])
			throw ts::Exception("Parameter slope should be 1 or equal to the dim dimension of the input parameter");
			
		output.resize(1);
		output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), stack.index(0)->sizes());

		return 1;
	}

	template<typename T>
	bool Prelu::prelu(ts::Stack &stack)
	{
		ts::Tensor& output_tensor = *stack.index(-1);
		auto output_shape = output_tensor.sizes();
		T* input_data = stack.index(0)->sync(memory_device()).data<T>();
		T* output_data = output_tensor.sync(memory_device()).data<T>();
		if (output_data == nullptr)
			return false;

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
						output_data[k + pre_offset] = std::max(input_data[k + pre_offset],T(0)) + val * std::min(input_data[k + pre_offset],T(0));
					}
				}
			}
		}
		else
		{
			float val = m_slope[0];
			for (int i = 0; i < output_tensor.count(); i++)
			{
				*output_data = std::max(*input_data, T(0)) + val * std::min(*input_data, T(0));
				input_data++;
				output_data++;
			}
		}
		return true;
	}
	TS_REGISTER_OPERATOR(Prelu, ts::CPU, "prelu")
}