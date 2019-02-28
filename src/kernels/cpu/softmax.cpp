#include <kernels/cpu/softmax.h>
#include <core/tensor_builder.h>
#include "backend/name.h"
#include "global/operator_factory.h"
#include <algorithm>
#include <math.h>

#include <kernels/common/simd.h>

namespace ts {
	namespace cpu {
		template<typename T>
		void cpu_softmax_compute_run(const Tensor &x, int m_dim, bool m_smooth, Tensor &out) {
			auto output_shape = out.sizes();

			int pre_num = 1;
			for (int i = 0; i < m_dim; i++) {
				pre_num *= output_shape[i];
			}
			int inner_num = 1;
			for (int i = m_dim + 1; i < output_shape.size(); i++) {
				inner_num *= output_shape[i];
			}

			int axis = output_shape[m_dim];

			auto device_type = x.device();
			const T *input_data = x.data<T>();
			T *output_data = out.data<T>();

			int count = out.count();
			memcpy(output_data, device_type, count * sizeof(T), input_data, device_type, count * sizeof(T));

			int scale_data_size = out.count() / axis;
			int denominator_data_size = scale_data_size;

			Shape scale_shape;
			scale_shape.resize(1);
			scale_shape[0] = scale_data_size;
			Tensor scale_tensor(MemoryDevice(CPU), out.dtype(), scale_shape);
			T *scale_data = scale_tensor.data<T>();

			Shape denominator_shape;
			denominator_shape.resize(1);
			denominator_shape[0] = denominator_data_size;
			Tensor denominator_tensor(MemoryDevice(CPU), out.dtype(), denominator_shape);
			T *denominator_data = denominator_tensor.data<T>();

			for (int i = 0; i < pre_num; i++) {
				std::memset(denominator_data, 0, scale_data_size * sizeof(T));
				if (m_smooth) {
					//Caculate max value
					memcpy(scale_data, device_type, inner_num * sizeof(T), input_data + i * axis * inner_num,
						   device_type, inner_num * sizeof(T));
					//std::memcpy(scale_data,input_data + i * axis * inner_num, inner_num * sizeof(T));
					for (int j = 0; j < axis; j++) {
						for (int k = 0; k < inner_num; k++) {
							scale_data[k] = std::max(scale_data[k],
													 input_data[i * axis * inner_num + j * inner_num + k]);
						}
					}
					//Caculate numerator and denominator
					for (int j = 0; j < axis; j++) {
						for (int k = 0; k < inner_num; k++) {
							output_data[i * axis * inner_num + j * inner_num + k] =
									output_data[i * axis * inner_num + j * inner_num + k] - scale_data[k];
							output_data[i * axis * inner_num + j * inner_num + k] = exp(
									output_data[i * axis * inner_num + j * inner_num + k]);
							denominator_data[k] += output_data[i * axis * inner_num + j * inner_num + k];
						}
					}
				} else {
					//Caculate numerator and denominator
					for (int j = 0; j < axis; j++) {
						for (int k = 0; k < inner_num; k++) {
							output_data[i * axis * inner_num + j * inner_num + k] = exp(
									output_data[i * axis * inner_num + j * inner_num + k]);
							denominator_data[k] += output_data[i * axis * inner_num + j * inner_num + k];
						}
					}
				}
				//Caculte output
				for (int j = 0; j < axis; j++) {
					for (int k = 0; k < inner_num; k++) {
						output_data[i * axis * inner_num + j * inner_num + k] =
								output_data[i * axis * inner_num + j * inner_num + k] / denominator_data[k];
					}
				}

			}
		}

        template<>
        void cpu_softmax_compute_run<float>(const Tensor &x, int m_dim, bool m_smooth, Tensor &out) {
            auto output_shape = out.sizes();

            int pre_num = 1;
            for (int i = 0; i < m_dim; i++) {
                pre_num *= output_shape[i];
            }
            int inner_num = 1;
            for (int i = m_dim + 1; i < output_shape.size(); i++) {
                inner_num *= output_shape[i];
            }

            int axis = output_shape[m_dim];

            auto device_type = x.device();
            const float *input_data = x.data<float>();
            float *output_data = out.data<float>();

            //memcpy(output_data, device_type, count * sizeof(float), input_data, device_type, count * sizeof(float));

            int scale_data_size = out.count() / axis;
            int denominator_data_size = scale_data_size;

            Shape scale_shape;
            scale_shape.resize(1);
            scale_shape[0] = scale_data_size;
            Tensor scale_tensor(MemoryDevice(CPU), out.dtype(), scale_shape);
            float *scale_data = scale_tensor.data<float>();

            Shape denominator_shape;
            denominator_shape.resize(1);
            denominator_shape[0] = denominator_data_size;
            Tensor denominator_tensor(MemoryDevice(CPU), out.dtype(), denominator_shape);
            float *denominator_data = denominator_tensor.data<float>();

            for (int i = 0; i < pre_num; i++) {
                std::memset(denominator_data, 0, scale_data_size * sizeof(float));
                int pre_offset = i * axis * inner_num;
                if (m_smooth) {         
                    //Caculate max value
                    memcpy(scale_data, device_type, inner_num * sizeof(float), input_data + pre_offset,
                        device_type, inner_num * sizeof(float));
                    //std::memcpy(scale_data,input_data + i * axis * inner_num, inner_num * sizeof(float));
                    for (int j = 0; j < axis; j++) {
                        int post_offset = j * inner_num;
                        for (int k = 0; k < inner_num - 3; k += 4) {
                            float *scale_temp = &scale_data[k];
                            float32x4 scale_data_x4(scale_temp);
                            float32x4 input_data_x4(&input_data[pre_offset + post_offset + k]);
                            scale_data_x4 = max_float32x4(scale_data_x4, input_data_x4);
                            scale_data_x4.store(scale_temp);
                        }
                        for (int k = inner_num/4*4; k < inner_num; k++) {
                            scale_data[k] = std::max(scale_data[k],
                                input_data[pre_offset + post_offset + k]);
                        }
                    }
                    //Caculate numerator and denominator
                    for (int j = 0; j < axis; j++) {
                        int post_offset = j * inner_num;
                        for (int k = 0; k < inner_num; k++) {
                            output_data[pre_offset + post_offset + k] = exp(
                                input_data[pre_offset + post_offset + k] - scale_data[k]);
                            denominator_data[k] += output_data[pre_offset + post_offset + k];
                        }
                    }
                }
                else {
                    //Caculate numerator and denominator
                    for (int j = 0; j < axis; j++) {
                        int post_offset = j * inner_num;
                        for (int k = 0; k < inner_num; k++) {
                            output_data[pre_offset + post_offset + k] = exp(
                                input_data[post_offset + post_offset + k]);
                            denominator_data[k] += output_data[pre_offset + post_offset + k];
                        }
                    }
                }
                //Caculte output
                for (int j = 0; j < axis; j++) {
                    int post_offset = j * inner_num;
                    for (int k = 0; k < inner_num - 3; k += 4) {
                        float *output_temp = &output_data[pre_offset + post_offset + k];
                        float32x4 output_data_x4(output_temp);
                        float32x4 denominator_data_x4(&denominator_data[k]);
                        output_data_x4 = output_data_x4 / denominator_data_x4;
                        output_data_x4.store(output_temp);
                    }
                    for (int k = inner_num/4*4; k < inner_num; k++)
                    {
                        output_data[pre_offset + post_offset + k] =
                            output_data[pre_offset + post_offset + k] / denominator_data[k];
                    }
                }

            }
        }

		void Softmax::softmax(const Tensor &x, int dim, bool smooth, Tensor &out) {
			// Notice: the all tensor' memory device are CPU, as given in running_memory_device
			DTYPE dtype = out.dtype();
			switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_softmax_compute_run<TYPE>(x, dim, smooth, out); break; }
				DECLARE_COMPUTE_RUN(INT8, int8_t);
				DECLARE_COMPUTE_RUN(UINT8, uint8_t);
				DECLARE_COMPUTE_RUN(INT16, int16_t);
				DECLARE_COMPUTE_RUN(UINT16, uint16_t);
				DECLARE_COMPUTE_RUN(INT32, int32_t);
				DECLARE_COMPUTE_RUN(UINT32, uint32_t);
				DECLARE_COMPUTE_RUN(INT64, int64_t);
				DECLARE_COMPUTE_RUN(UINT64, uint64_t);
				DECLARE_COMPUTE_RUN(FLOAT32, float);
				DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
				default: {
					TS_LOG_ERROR << this->op() << " not support this data type: " << dtype << eject;
					break;
				}
			}
		}
	}
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Softmax, ts::CPU, name::layer::softmax())
