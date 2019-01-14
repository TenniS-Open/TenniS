#ifndef TS_KERNELS_POOLONG2D_H
#define TS_KERNELS_POOLONG2D_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>

#include "backend/common_structure.h"

namespace ts {
	class Pooling2d : public ts::Operator {
	public:
		using supper = ts::Operator;
		Pooling2d() {
			field("format", REQUIRED);
			field("type", REQUIRED);
			field("padding", REQUIRED);
			Tensor default_padding_type(INT32, { 1 });
			default_padding_type.data<int>()[0] = 0;
			field("padding_type", OPTIONAL, default_padding_type);
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

		template<typename T>
		bool pooling(ts::Stack &stack);

		template<typename T>
		bool max_pooling(T* input_data, T* output_data, Shape& input_shape, Shape& output_shape, KSize2D& ksize, Stride2D& stride);

		template<typename T>
		bool average_pooling(T* input_data, T* output_data, Shape& input_shape, Shape& output_shap, KSize2D& ksize, Stride2D& stride);

	private:

		Padding2D m_padding;
		std::string m_format;
		POOLINGTYPE m_pooling_type;
		PDDINGTYPE m_padding_type;
		KSize2D m_ksize;
		Stride2D m_stride;
	};
	TS_REGISTER_OPERATOR(Pooling2d, ts::CPU, "pooling2d")
}

#endif //TS_KERNELS_POOLONG2D_H