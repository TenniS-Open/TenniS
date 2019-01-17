#include <kernels/cpu/pooling2d_v2.h>
#include <core/tensor_builder.h>
#include "backend/name.h"
#include "runtime/stack.h"
#include "runtime/instruction/instruction_factory.h"
#include "utils/assert.h"

#include "global/memory_device.h"
#include "backend/common_function.h"

namespace ts {

	Pooling2dV2::Pooling2dV2()
        :m_padding(0,0,0,0)
        , m_ksize(0,0)
        , m_stride(0,0)
	{
		field(name::format, REQUIRED);
		field(name::type, REQUIRED);
		//field(name::padding, REQUIRED);
		Tensor default_padding_type = tensor::from<int32_t>(PDDINGTYPE::black);
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
		m_format = tensor::to_string(format_tensor);

		auto pooling_type_tensor = this->get(name::type);
		m_operator->set(name::type, pooling_type_tensor);

		//auto padding_tensor = this->get(name::padding);
		//m_operator->set(name::padding, padding_tensor);

		auto padding_tensor = this->get(name::padding_type);
		m_operator->set(name::padding_type, padding_tensor);
	}

	int Pooling2dV2::run(ts::Stack &stack)
	{
		TS_AUTO_CHECK(stack.size() != 0);
		TS_AUTO_CHECK(stack.size() == 4 && stack.index(0)->dims() == 4);
		TS_AUTO_CHECK(stack.index(0)->dtype() == FLOAT32 || stack.index(0)->dtype() == FLOAT64);

		//auto controller = std::make_shared<DynamicMemoryController>(MemoryDevice(CPU));
        Padding2D padding;
        KSize2D ksize;
        Stride2D stride;

		auto padding_tensor = tensor::cast(INT32, *stack.index(1));
        auto ksize_tensor = tensor::cast(INT32, *stack.index(2));
        auto stride_tensor = tensor::cast(INT32, *stack.index(3));
        
        if (m_format == name::NCHW)
        {
            padding.top = padding_tensor.data<int32_t>()[4];
            padding.bottom = padding_tensor.data<int32_t>()[5];
            padding.left = padding_tensor.data<int32_t>()[6];
            padding.right = padding_tensor.data<int32_t>()[7];
            ksize.height = ksize_tensor.data<int32_t>()[2];
            ksize.width = ksize_tensor.data<int32_t>()[3];
            stride.height = stride_tensor.data<int32_t>()[2];
            stride.width = stride_tensor.data<int32_t>()[3];
        }
        else
        {
            padding.top = padding_tensor.data<int32_t>()[2];
            padding.bottom = padding_tensor.data<int32_t>()[3];
            padding.left = padding_tensor.data<int32_t>()[4];
            padding.right = padding_tensor.data<int32_t>()[5];
            ksize.height = ksize_tensor.data<int32_t>()[1];
            ksize.width = ksize_tensor.data<int32_t>()[2];
            stride.height = stride_tensor.data<int32_t>()[1];
            stride.width = stride_tensor.data<int32_t>()[2];
        }

        if (!m_init_pooling2d_flag || !check_equale_input_param(padding, ksize, stride))
        {
            m_operator->set(name::padding, tensor::clone(padding_tensor.dtype(), padding_tensor));
            m_operator->set(name::ksize, tensor::clone(ksize_tensor.dtype(), ksize_tensor));
            m_operator->set(name::stride, tensor::clone(stride_tensor.dtype(), stride_tensor));
            m_operator->init();
            m_init_pooling2d_flag = true;
        }

		stack.pop();
		stack.pop();
		stack.pop();

		RunOperator(m_operator, stack, 1);

		return 1;
	}

	int Pooling2dV2::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output)
	{
        TS_AUTO_CHECK(stack.size() == 1 && stack.index(0)->dims() == 4);
        TS_AUTO_CHECK(stack.index(0)->dtype() == FLOAT32 || stack.index(0)->dtype() == FLOAT64);

        Size2D input_size;

        if (m_format == name::NCHW)
        {
            input_size.height = stack.index(0)->sizes()[2];
            input_size.width = stack.index(0)->sizes()[3];
        }
        else
        {
            input_size.height = stack.index(0)->sizes()[1];
            input_size.width = stack.index(0)->sizes()[2];
        }

        Size2D output_size;
        output_size = pooling2d_forward(input_size, m_padding, m_ksize, m_stride);

        Shape output_shape(stack.index(0)->sizes());
        if (m_format == name::NCHW)
        {
            output_shape[2] = output_size.height;
            output_shape[3] = output_size.width;
        }
        else
        {
            output_shape[1] = output_size.height;
            output_shape[2] = output_size.width;
        }
        output.resize(1);
        output[0] = ts::Tensor::Prototype(stack.index(0)->dtype(), output_shape);
        return 1;
	}

    bool Pooling2dV2::check_equale_input_param(const Padding2D& padding, const KSize2D& ksize, const Stride2D& stride)
    {
        if (padding.bottom != m_padding.bottom || padding.top != m_padding.top || padding.left != m_padding.left || padding.right != m_padding.right)
            return false;
        if (ksize.height != m_ksize.height || ksize.width != m_ksize.width)
            return false;
        if (stride.height != m_ksize.height || stride.width != m_ksize.width)
            return false;
        return true;
    }
}

using namespace ts;
TS_REGISTER_OPERATOR(Pooling2dV2, ts::CPU, name::layer::pooling2d_v2())