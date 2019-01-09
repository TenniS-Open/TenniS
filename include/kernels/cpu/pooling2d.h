#ifndef TS_KERNELS_POOLONG2D_H
#define TS_KERNELS_POOLONG2D_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>

#include <string>
#include <algorithm>

namespace ts {
	class Pooling2d : public ts::Operator {
	public:
		using supper = ts::Operator;
		Pooling2d() {
			field("format", REQUIRED);
			field("type", REQUIRED);
			field("padding", REQUIRED);
			field("padding_type", OPTIONAL);
			//Tensor default_pooling_value(FLOAT32, { 1 });
			//default_pooling_value.data<int>()[0] = 0;
			//field("padding_value", OPTIONAL, default_pooling_value);
			field("ksize", REQUIRED);
			field("stride", REQUIRED);
		}

		enum POOLINGTYPE {
			max,
			avg
		};

		enum PDDINGTYPE {
			black,
			copy,
			loop
		};

		virtual void init();
		virtual int run(ts::Stack &stack);
		virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output);

	private:
		void caculate_pool_size(int& output_h, int& output_w);

		template<typename T>
		bool pooling(ts::Stack &stack);

		template<typename T>
		bool max_pooling(T* input_data, T* output_data);

		template<typename T>
		bool average_pooling(T* input_data, T* output_data);

	private:
		int m_batch_num, m_channel, m_input_h, m_input_w;
		int m_pad_h_up, m_pad_h_down, m_pad_w_left, m_pad_w_right, m_kernel_h, m_kernel_w, m_stride_h, m_stride_w;
		int m_output_h, m_output_w;
		std::string m_format;
		POOLINGTYPE m_pooling_type;
		PDDINGTYPE m_padding_type;
	};
	TS_REGISTER_OPERATOR(Pooling2d, ts::CPU, "pooling2d")
}

#endif //TS_KERNELS_POOLONG2D_H