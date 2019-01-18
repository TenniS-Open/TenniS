#ifndef TS_KERNELS_PREWHITEN_H
#define TS_KERNELS_PREWHITEN_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>

namespace ts {
	class PreWhiten : public ts::Operator {
	public:
		using supper = ts::Operator;
		PreWhiten() {}
		virtual void init();
		virtual int run(ts::Stack &stack);
		virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output);

	private:
		template<typename T>
		bool pre_whiten(ts::Stack &stack);
	};

	//TS_REGISTER_OPERATOR(PreWhiten, ts::CPU, "prewhiten")
}



#endif //TS_KERNELS_PREWHITEN_H