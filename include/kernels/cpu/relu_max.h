#ifndef TS_KERNELS_RELUMAX_H
#define TS_KERNELS_RELUMAX_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>

namespace ts {
	class ReluMax : public ts::Operator {
	public:
		using supper = ts::Operator;
		ReluMax() {
			field("max", REQUIRED);
		}
		virtual void init();
		virtual int run(ts::Stack &stack);
		virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output);

	private:
		template<typename T>
		bool relu_max(ts::Stack &stack);

	private:
		float m_max;
	};
	//TS_REGISTER_OPERATOR(ReluMax, ts::CPU, "relu_max")
}



#endif //TS_KERNELS_RELUMAX_H