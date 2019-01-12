#ifndef TS_KERNELS_SIGMOID_H
#define TS_KERNELS_SIGMOID_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>

namespace ts {
	class Sigmoid : public ts::Operator {
	public:
		using supper = ts::Operator;
		Sigmoid() {
			
		}
		virtual void init();
		virtual int run(ts::Stack &stack);
		virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output);

	private:
		template<typename T>
		bool sigmoid(ts::Stack &stack);
	};
	//TS_REGISTER_OPERATOR(Sigmoid, ts::CPU, "sigmoid")
}



#endif //TS_KERNELS_SIGMOID_H