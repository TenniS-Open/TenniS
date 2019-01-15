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

		Operator::shared m_operator;

	};
	//TS_REGISTER_OPERATOR(Pooling2dV2, ts::CPU, "pooling2d_v2")
}

#endif //TS_KERNELS_POOLONG2DV2_H