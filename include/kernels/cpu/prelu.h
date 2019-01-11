#ifndef TS_KERNELS_PRELU_H
#define TS_KERNELS_PRELU_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>

#include <vector>

namespace ts {
	class Prelu : public ts::Operator {
	public:
		using supper = ts::Operator;
		Prelu() {
			field("dim", OPTIONAL);
			field("slope", REQUIRED);
			m_dim = -1;
		}
		virtual void init();
		virtual int run(ts::Stack &stack);
		virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output);

	private:
		template<typename T>
		bool prelu(ts::Stack &stack);
	private:
		int m_dim;
		std::vector<float> m_slope;
	};

	//TS_REGISTER_OPERATOR(Prelu, ts::CPU, "prelu")
}



#endif //TS_KERNELS_PRELU_H