#include <kernels/cpu/concat.h>
#include <core/tensor_builder.h>

#include "backend/name.h"
#include "core/memory.h"

namespace ts {

	Concat::Concat() :m_dim(-1)
	{
		field(name::dim, REQUIRED);
	}

	void Concat::init()
	{
		supper::init();

		auto dim_tensor = get(name::dim);
		m_dim = tensor::to_int(dim_tensor);
		
		TS_AUTO_CHECK(m_dim > 0);
	}

	int Concat::run(ts::Stack &stack)
	{
		int input_num = stack.size();

		std::vector<ts::Tensor::Prototype> output;

		this->infer(stack, output);
		stack.push(output[0], memory_device());

		auto dtype = stack.index(0)->dtype();
		bool flag;
		switch (dtype)
		{
			case ts::INT8:
			{
				flag = concat<char>(stack, input_num);
				break;
			}
			case ts::UINT8:
			{
				flag = concat<unsigned char>(stack, input_num);
				break;
			}
			case ts::INT16:
			{
				flag = concat<short>(stack, input_num);
				break;
			}
			case ts::UINT16:
			{
				flag = concat<unsigned short>(stack, input_num);
				break;
			}
			case ts::INT32:
			{
				flag = concat<int>(stack, input_num);
				break;
			}
			case ts::UINT32:
			{
				flag = concat<unsigned int>(stack, input_num);
				break;
			}
			case ts::FLOAT32:
			{
				flag = concat<float>(stack, input_num);
				break;
			}
			case ts::FLOAT64:
			{
				flag = concat<double>(stack, input_num);
				break;
			}
			default:break;
		}
		return 1;
	}

	int Concat::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output)
	{
		int input_num = stack.size();

		TS_AUTO_CHECK(input_num != 0);
		TS_AUTO_CHECK(stack.index(0)->dtype() == FLOAT32 || stack.index(0)->dtype() == FLOAT64);

		if (input_num == 1)
		{
			output.resize(1);
			output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), stack.index(0)->sizes());
			return 1;
		}

		auto dtype = stack.index(0)->dtype();

		for (int i = 1; i < input_num; i++)
		{
			TS_CHECK(stack.index(i)->dtype() == dtype) << "Can not concat on different data type" << ts::eject;
		}
		
		Shape output_shape(stack.index(0)->sizes());

		TS_CHECK(m_dim < output_shape.size()) << "Concat dim should be less than input dim!" << ts::eject;

		int num_dims = output_shape.size();
		int concat_dim_output_num = output_shape[m_dim];

		for (int i = 1; i < input_num; i++)
		{
			auto shape = stack.index(i)->sizes();
			TS_CHECK(shape.size() == num_dims) << "All inputs must have the same dims!" << ts::eject;

			for (int j = 0; j < shape.size(); j++)
			{
				if (j == m_dim)
					continue;
				TS_CHECK(shape[j] == output_shape[j]) << "All inputs must have the same shape, except at concat_axis!" << ts::eject;
			}
			concat_dim_output_num += shape[m_dim];
		}
		
		output_shape[m_dim] = concat_dim_output_num;

		output.resize(1);
		output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), output_shape);

		return 1;
	}

	template<typename T>
	bool Concat::concat(ts::Stack &stack,int input_num)
	{
		auto input_shape = stack.index(0)->sizes();
		ts::Tensor& output_tensor = *stack.index(-1);
		auto output_shape = output_tensor.sizes();
		T* output_data = output_tensor.sync(MemoryDevice(CPU)).data<T>();

		int num_concats = 1;              
		int input_concat_size = 1;      
		int output_concat_axis = output_shape[m_dim];
		
		for (int i = 0; i < m_dim; i++)
		{
				num_concats *= input_shape[i];
		}

		for (int i = m_dim + 1; i < input_shape.size(); i++)
		{
			input_concat_size *= input_shape[i];
		}

		int offset_concat_axis = 0;
		for (int i = 0; i < input_num; i++)
		{
			const T* input_data = stack.index(i)->sync(memory_device()).data<T>();
			int input_concat_axis = stack.index(i)->sizes()[m_dim];
			for (int j = 0; j < num_concats; j++)
			{
				memcpy(output_data + (j * output_concat_axis + offset_concat_axis)* input_concat_size,MemoryDevice(CPU), input_concat_axis * input_concat_size * sizeof(T),
					input_data + j * input_concat_axis * input_concat_size, MemoryDevice(CPU), input_concat_axis * input_concat_size * sizeof(T));
			}
			offset_concat_axis += input_concat_axis;
		}
		//int output_index = 0;
		//for (int i = 0; i < input_num; i++)
		//{
		//	const T* input_data = stack.index(i)->sync(memory_device()).data<T>();
		//	int input_concat_axis = stack.index(i)->sizes()[m_dim];
		//	int input_index = 0;
		//	for (int j = 0; j < num_concats; j++)
		//	{
		//		int output_index_temp = output_index;
		//		int input_index_temp = input_index;
		//		for (int num = 0; num < input_concat_axis * input_concat_size; num++)
		//		{
		//			output_data[output_index_temp++] = input_data[input_index_temp++];
		//		}
		//		input_index += input_concat_axis * input_concat_size;
		//		output_index += output_concat_axis * input_concat_size;
		//	}
		//	offset_concat_axis += input_concat_axis;
		//	output_index = offset_concat_axis * input_concat_size;
		//}

		return true;
	}
}

using namespace ts;
TS_REGISTER_OPERATOR(Concat, ts::CPU, name::layer::concat())