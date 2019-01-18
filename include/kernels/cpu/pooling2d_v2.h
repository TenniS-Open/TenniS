#ifndef TS_KERNELS_POOLONG2DV2_H
#define TS_KERNELS_POOLONG2DV2_H

#include <global/operator_factory.h>
#include <core/tensor.h>

#include "backend/common_structure.h"

namespace ts {
	class Pooling2dV2 : public ts::Operator {
	public:
		using self = Pooling2dV2;
		using supper = ts::Operator;

		Pooling2dV2();

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
		bool check_equale_input_param(const Padding2D& padding,const KSize2D& ksize,const Stride2D& stride);
	private:

		Operator::shared m_operator;
		Padding2D m_padding;
		KSize2D m_ksize;
		Stride2D m_stride;
		std::string m_format;
		bool m_init_pooling2d_flag = false;

	};
	//TS_REGISTER_OPERATOR(Pooling2dV2, ts::CPU, "pooling2d_v2")
}

#endif //TS_KERNELS_POOLONG2DV2_H