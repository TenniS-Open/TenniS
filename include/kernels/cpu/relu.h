#ifndef TS_KERNELS_RELU_H
#define TS_KERNELS_RELU_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>

namespace ts {
	class Relu : public ts::Operator {
	public:
		using supper = ts::Operator;
		Relu() {}
		virtual void init();
		virtual int run(ts::Stack &stack);
		virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output);

	private:
		template<typename T>
		bool relu(ts::Stack &stack);
	};

	//TS_REGISTER_OPERATOR(Relu, ts::CPU, "relu")
}



#endif //TS_KERNELS_RELU_H