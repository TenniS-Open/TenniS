#ifndef TS_KERNELS_SOFTMAX_H
#define TS_KERNELS_SOFTMAX_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>

#include <vector>

namespace ts {
	class Softmax : public ts::Operator {
	public:
		using supper = ts::Operator;
		Softmax() {
			field("dim", REQUIRED);
		}
		virtual void init();
		virtual int run(ts::Stack &stack);
		virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output);

	private:
		template<typename T>
		bool softmax(ts::Stack &stack);
	private:

	private:
		int m_dim;
	};

	//TS_REGISTER_OPERATOR(Softmax, ts::CPU, "softmax")
}



#endif //TS_KERNELS_SOFTMAX_H