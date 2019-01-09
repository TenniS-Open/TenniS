#include <kernels/cpu/pooling2d.h>
#include <core/tensor_builder.h>

namespace ts {

	void Pooling2d::init()
	{
		supper::init();

		if (!has("format") || !has("type") || !has("padding") || !has("ksize") || !has("stride"))
			throw ts::Exception("Missing patameter!");

		ts::Tensor& fomat_param = get("format");
		if (fomat_param.dtype() != CHAR8)
			throw ts::Exception("The Fomat parameter type must be string!");
		m_format = tensor::to_string(fomat_param);
		//m_format = "NCHW";

		ts::Tensor& padding_param = get("padding");
		//if (padding_param.dims() != 4)
		//	throw ts::Exception("The Padding parameter must be four-dimensional!");
		if (padding_param.count() != 8)
			throw ts::Exception("The Padding parameter must have eight!");
		auto padding_memory = padding_param.sync(memory_device());
		
		ts::Tensor& padding_type_param = get("padding_type");
		m_padding_type = (PDDINGTYPE)tensor::to_int(padding_type_param);

		//ts::Tensor& padding_value_param = get("padding_value");
		//if (padding_value_param.dtype() != FLOAT32)
		//	throw ts::Exception("The Padding value parameter type must be FLOAT32!");  //default type

		ts::Tensor& ksize_param = get("ksize");

		ts::Tensor& stide_param = get("stride");

		if (m_format == "NCHW")
		{
			m_pad_h_up = padding_memory.data<int>()[4];
			m_pad_h_down = padding_memory.data<int>()[5];
			m_pad_w_left = padding_memory.data<int>()[6];
			m_pad_w_right = padding_memory.data<int>()[7];
			m_kernel_h = ksize_param.data<int>()[2];
			m_kernel_w = ksize_param.data<int>()[3];
			m_stride_h = stide_param.data<int>()[2];
			m_stride_w = stide_param.data<int>()[3];
			if (padding_memory.data<int>()[0] != 0 || padding_memory.data<int>()[1] != 0 || padding_memory.data<int>()[2] != 0 || padding_memory.data<int>()[3] != 0)
				throw ts::Exception("The Padding value parameter error!");

			if (ksize_param.sync(memory_device()).data<int>()[0] != 0 || ksize_param.sync(memory_device()).data<int>()[1] != 0)
				throw ts::Exception("The ksize parameter error!");
			if (ksize_param.sync(memory_device()).data<int>()[2] < 0 || ksize_param.sync(memory_device()).data<int>()[3] < 0)
				throw ts::Exception("The ksize parameter must be greater than or equal to 0!");

			if (stide_param.sync(memory_device()).data<int>()[0] != 0 || stide_param.sync(memory_device()).data<int>()[1] != 0)
				throw ts::Exception("The stride parameter error!");
			if (stide_param.sync(memory_device()).data<int>()[2] < 0 || stide_param.sync(memory_device()).data<int>()[3] < 0)
				throw ts::Exception("The stride parameter must be greater than or equal to 0!");
		}
		else if (m_format == "NHWC")
		{
			m_pad_h_up = padding_memory.data<int>()[2];
			m_pad_h_down = padding_memory.data<int>()[3];
			m_pad_w_left = padding_memory.data<int>()[4];
			m_pad_w_right = padding_memory.data<int>()[5];
			m_kernel_h = ksize_param.data<int>()[1];
			m_kernel_w = ksize_param.data<int>()[2];
			m_stride_h = stide_param.data<int>()[1];
			m_stride_w = stide_param.data<int>()[2];
			if (padding_memory.data<int>()[0] != 0 || padding_memory.data<int>()[1] != 0 || padding_memory.data<int>()[6] != 0 || padding_memory.data<int>()[7] != 0)
				throw ts::Exception("The Padding value parameter error!");

			if (ksize_param.sync(memory_device()).data<int>()[0] != 0 || ksize_param.sync(memory_device()).data<int>()[3] != 0)
				throw ts::Exception("The ksize parameter error!");
			if (ksize_param.sync(memory_device()).data<int>()[1] < 0 || ksize_param.sync(memory_device()).data<int>()[2] < 0)
				throw ts::Exception("The ksize parameter must be greater than or equal to 0!");

			if (stide_param.sync(memory_device()).data<int>()[0] != 0 || stide_param.sync(memory_device()).data<int>()[3] != 0)
				throw ts::Exception("The stride parameter error!");
			if (stide_param.sync(memory_device()).data<int>()[1] < 0 || stide_param.sync(memory_device()).data<int>()[2] < 0)
				throw ts::Exception("The stride parameter must be greater than or equal to 0!");
		}
		else
		{
			throw ts::Exception("The Format parameter must be NCHW or NHWC!");
		}

		m_pooling_type = (POOLINGTYPE)tensor::to_int(get("type"));
	}

	int Pooling2d::run(ts::Stack &stack)
	{
		int input_num = stack.size();
		std::vector<ts::Tensor::Prototype> output;

		this->infer(stack, output);
		stack.push(output[0], memory_device());

		int type_len = ts::type_bytes(stack.index(0)->dtype());
		if (type_len == 1)
		{
			pooling<unsigned char>(stack);
		}
		else if (type_len == 2)
		{
			pooling<unsigned short>(stack);
		}
		else if (type_len == 4)
		{
			pooling<float>(stack);
		}
		else if (type_len == 8)
		{
			pooling<double>(stack);
		}

		return 1;
	}

	int Pooling2d::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output)
	{
		if (stack.size() == 0)
			throw ts::Exception("Can not pooling on empty inputs");

		if (stack.size() != 1 || stack.index(0)->dims() != 4)
			throw ts::Exception("Input parameter is invalid");

		if (m_format == "NCHW")
		{
			m_batch_num = stack.index(0)->sizes()[0];
			m_channel = stack.index(0)->sizes()[1];
			m_input_h = stack.index(0)->sizes()[2];
			m_input_w = stack.index(0)->sizes()[3];
		}
		else
		{
			m_batch_num = stack.index(0)->sizes()[0];
			m_channel = stack.index(0)->sizes()[3];
			m_input_h = stack.index(0)->sizes()[1];
			m_input_w = stack.index(0)->sizes()[2];
		}

		caculate_pool_size(m_output_h, m_output_w);

		Shape output_shape(stack.index(0)->sizes());
		if (m_format == "NCHW")
		{
			output_shape[2] = m_output_h;
			output_shape[3] = m_output_w;
		}
		else
		{
			output_shape[1] = m_output_h;
			output_shape[2] = m_output_w;
		}
		output.resize(1);
		output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), output_shape);
		return 1;
	}

	void Pooling2d::caculate_pool_size(int& output_h, int& output_w)
	{
		if (m_pooling_type == max)
		{
			output_h = ceil((m_input_h + m_pad_h_up + m_pad_h_down - m_kernel_h) / static_cast<float>(m_stride_h) + 1);
			output_w = ceil((m_input_w + m_pad_w_left + m_pad_w_right - m_kernel_w) / static_cast<float>(m_stride_w) + 1);
		}
		else
		{
			output_h = floor((m_input_h + m_pad_h_up + m_pad_h_down - m_kernel_h) / static_cast<float>(m_stride_h) + 1);
			output_w = floor((m_input_w + m_pad_w_left + m_pad_w_right - m_kernel_w) / static_cast<float>(m_stride_w) + 1);
		}
	}

	template<typename T>
	bool Pooling2d::pooling(ts::Stack &stack)
	{
		T* input_data = stack.index(0)->sync(memory_device()).data<T>();
		ts::Tensor& output_tensor = *stack.index(-1);
		T* output_data = output_tensor.sync(memory_device()).data<T>();
		if (input_data == nullptr || output_data == nullptr)
			return false;
		bool flag;
		if (m_pooling_type == max)
		{
			flag = max_pooling<T>(input_data, output_data);
		}
		else
		{
			flag = average_pooling<T>(input_data, output_data);
		}
		return flag;
	}

	template<typename T>
	bool Pooling2d::max_pooling(T* input_data, T* output_data)
	{
		int input_channel_size = m_input_h * m_input_w;
		int output_channel_size = m_output_h * m_output_w;
		for (int n = 0; n < m_batch_num; n++)
		{
			for (int c = 0; c< m_channel; c++)
			{
				for (int oh = 0; oh < m_output_h; oh++)
				{
					for (int ow = 0; ow < m_output_w; ow++)
					{
						int ihStart = oh * m_stride_h - m_pad_h_up;
						int iwStart = ow * m_stride_w - m_pad_w_left;
						int ihEnd = std::min<T>(ihStart + m_kernel_h, m_input_h);
						int iwEnd = std::min<T>(iwStart + m_kernel_w, m_input_w);
						ihStart = std::max<T>(ihStart, 0);
						iwStart = std::max<T>(iwStart, 0);
						int outIndex = oh * m_output_w + ow;
						T maxVlue = 0;
						//int count = 0;
						for (int ih = ihStart; ih < ihEnd; ih++)
						{
							for (int iw = iwStart; iw < iwEnd; iw++)
							{
								int input_index = ih * m_input_w + iw;
								if (input_data[input_index] > maxVlue)
								{
									maxVlue = input_data[input_index];
								}
							}
							//count++;
						}
						output_data[outIndex] = maxVlue;
						//if (count == m_kernel_h * m_kernel_w)
						//	output_data[outIndex] = maxVlue;
						//else
						//	output_data[outIndex] = std::max<T>(maxVlue, padding_value);
					}
				}
				input_data += input_channel_size;
				output_data += output_channel_size;
			}
		}
		return true;
	}

	template<typename T>
	bool Pooling2d::average_pooling(T* input_data, T* output_data)
	{
		int input_channel_size = m_input_h * m_input_w;
		int output_channel_size = m_output_h * m_output_w;
		for (int n = 0; n < m_batch_num; n++)
		{
			for (int c = 0; c< m_channel; c++)
			{
				for (int oh = 0; oh < m_output_h; oh++)
				{
					for (int ow = 0; ow < m_output_w; ow++)
					{
						int ihStart = oh * m_stride_h - m_pad_h_up;
						int iwStart = ow * m_stride_w - m_pad_w_left;
						int ihEnd = std::min<T>(ihStart + m_kernel_h, m_input_h);
						int iwEnd = std::min<T>(iwStart + m_kernel_w, m_input_w);
						ihStart = std::max<T>(ihStart, 0);
						iwStart = std::max<T>(iwStart, 0);
						int outIndex = oh * m_output_w + ow;
						T sumValue = 0.0;
						int count = 0;
						for (int ih = ihStart; ih < ihEnd; ih++)
						{
							for (int iw = iwStart; iw < iwEnd; iw++)
							{
								int input_index = ih * m_input_w + iw;
								sumValue += input_data[input_index];
								count++;
							}
						}
						if (count == 0)
							output_data[outIndex] = 0;
						else
							output_data[outIndex] = sumValue / count;
						//if (count == 0)
						//	output_data[outIndex] = 0;
						//else if (count == m_kernel_h * m_kernel_w)
						//	output_data[outIndex] = sumValue / count;
						//else
						//	output_data[outIndex] = (sumValue + (m_kernel_h * m_kernel_w - count) * padding_value) / (m_kernel_h * m_kernel_w);
					}
				}
				input_data += input_channel_size;
				output_data += output_channel_size;
			}
		}
		return true;
	}

	//TS_REGISTER_OPERATOR(Pooling2d, ts::CPU, "pooling2d")

}