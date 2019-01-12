#ifndef TS_KERNELS_CONCAT_H
#define TS_KERNELS_CONCAT_H

#include <global/operator_factory.h>
#include <core/tensor.h>
#include <runtime/stack.h>

namespace ts {

	class Concat : public ts::Operator {
	public:
		using supper = ts::Operator;
		Concat():m_dim(-1) 
		{
			field("dim", REQUIRED);
		}
		virtual void init();
		virtual int run(ts::Stack &stack);
		virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output);
	private:

		template<typename T>
		bool concat(ts::Stack &stack, int input_num);

	private:
		int m_dim;
	};

	//TS_REGISTER_OPERATOR(Concat, ts::CPU, "concat")
}


#endif //TS_KERNELS_CONCAT_H