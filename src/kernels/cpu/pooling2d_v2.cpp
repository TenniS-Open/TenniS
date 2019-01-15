#include <kernels/cpu/pooling2d_v2.h>
#include <core/tensor_builder.h>
#include "backend/common_function.h"
#include "backend/name.h"
#include "runtime/stack.h"
#include "runtime/instruction/instruction_factory.h"
#include "utils/assert.h"

#include "global/memory_device.h"

namespace ts {

	Pooling2dV2::Pooling2dV2()
	{
		field(name::format, REQUIRED);
		field(name::type, REQUIRED);
		field(name::padding, REQUIRED);
		Tensor default_padding_type(INT32, { 1 });
		default_padding_type.data<int>()[0] = 0;
		field(name::padding_type, OPTIONAL, default_padding_type);
	}

	void Pooling2dV2::init()
	{
		supper::init();

		auto creator = OperatorCreator::Query(ts::CPU, name::layer::pooling2d());
		if (creator == nullptr) {
			// step 2.x: try find operator on memory device, if computing device failed
			auto memory_device = ComputingMemory::Query(this->computing_device());
			creator = OperatorCreator::Query(memory_device.type(), name::layer::pooling2d());
		}

		if (creator == nullptr) {
			// step 2.y: try find operator on CPU version
			if (this->computing_device().type() != CPU) {
				creator = OperatorCreator::Query(CPU, name::layer::pooling2d());
			}
		}

		if (creator == nullptr) TS_LOG_ERROR << "Not supported operator " << name::layer::pooling2d() << eject;

		m_operator = creator();

		auto format_tensor = this->get(name::format);
		m_operator->set(name::format, format_tensor);

		auto pooling_type_tensor = this->get(name::type);
		m_operator->set(name::type, pooling_type_tensor);

		auto padding_tensor = this->get(name::padding);
		m_operator->set(name::padding, padding_tensor);

		if (has("padding_type"))
		{
			auto padding_tensor = this->get(name::padding_type);
			m_operator->set(name::padding_type, padding_tensor);
		}

	}

	int Pooling2dV2::run(ts::Stack &stack)
	{
		TS_AUTO_CHECK(stack.size() != 0);
		TS_AUTO_CHECK(stack.size() == 3 && stack.index(0)->dims() == 4);
		TS_AUTO_CHECK(stack.index(0)->dtype() == FLOAT32 || stack.index(0)->dtype() == FLOAT64);

		auto ksize_tensor = tensor::cast(INT32, *stack.index(1));
		m_operator->set(name::ksize, ksize_tensor);

		auto stride_tensor = tensor::cast(INT32, *stack.index(2));
		m_operator->set(name::stride, stride_tensor);

		m_operator->init();

		//stack.pop();
		//stack.pop();

		try
		{
			return m_operator->run(stack);
		}
		catch (const Exception &e) {
			std::cout << e.what() << std::endl;
			return -1;
		}
	}

	int Pooling2dV2::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output)
	{
		try
		{
			return m_operator->infer(stack, output);
		}
		catch (const Exception &e) {
			std::cout << e.what() << std::endl;
			return -1;
		}
	}
}

using namespace ts;
TS_REGISTER_OPERATOR(Pooling2dV2, ts::CPU, "pooling2d_v2")