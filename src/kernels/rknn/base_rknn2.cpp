//
// Created by kier on 2020/6/5.
//

#include "base_rknn2.h"

#include "backend/name.h"
#include "core/tensor.h"
#include "core/tensor_builder.h"
#include "utils/need.h"

#include <fstream>
#include <core/device_context.h>

#include "frontend/intime.h"

namespace ts {
    namespace base {
        Rknn2::Rknn2() {
            field("input_count", REQUIRED);
            field("output_count", REQUIRED);

            field("onnx_file", OPTIONAL);   // not used
            field("rknn_file", OPTIONAL);   // not used
            field("rknn_buffer", OPTIONAL);

            field("input_shapes", OPTIONAL);
            field("output_shapes", OPTIONAL);

            field("format", REQUIRED);
        }

        Rknn2::~Rknn2() {
            if (m_ctx) {
                int ret = m_rknn->destroy(m_ctx);
                if (ret < 0) {
                    TS_LOG_ERROR << this->op() << " [RKNN] rknn_destroy " << m_ctx << " failed!"
                                 << " Error(" << ret << "): " << get_rknn_error_str(ret) << ".";
                } else {
                    TS_LOG_INFO << this->op() << " [RKNN] rknn_destroy " << m_ctx << " succeed!";
                }
            }
            delete m_rknn;
        }

        static std::string shape_string(const uint32_t *rknn_data, uint32_t rknn_size) {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < rknn_size; ++i) {
                if (i) oss << ", ";
                oss << (rknn_data[i] >= 0 ? std::to_string(rknn_data[i]) : "?");
            }
            oss << "]";
            return oss.str();
        }

        void Rknn2::init() {
            supper::init();

            m_rknn = new RknnDll();

            // get device id
            // auto &context = ctx::ref<DeviceContext>();
            // auto device_id = context.computing_device.id();
            bool device_found = false;
            // char *device_name;

            // rknn_devices_id found_devices;
            // std::memset(&found_devices, 0, sizeof(found_devices));
            // auto found_error = rknn_find_devices(&found_devices);
            // if (found_error < 0){
            //     TS_LOG_ERROR << this->op() << " [RKNN] rknn_find_devices failed!"
            //                  << " Error(" << found_error << "): " << get_rknn_error_str(found_error) << ".";
            // } else if (device_id == 0) {
            //     // do not use multi device API
            // } else {
            //     device_found = true;
            //     if (device_id >= found_devices.n_devices) {
            //         TS_LOG_ERROR << this->op() << " [RKNN] device rknn:" << device_id << " no found. Only " << found_devices.n_devices << " device(s) found." << eject;
            //     }
            //     device_name = found_devices.ids[device_id];
            //     TS_LOG_ERROR << this->op() << " [RKNN] found device rknn:" << device_id
            //         << " {type=" << found_devices.types[device_id] << ", id=" << found_devices.ids[device_id] << "}";
            // }


            auto format = tensor::to_string(get("format"));
            if (format == name::NCHW) {
                m_data_format = DataFormat::DATA_NCHW;
            } else if (format == name::NHWC) {
                m_data_format = DataFormat::DATA_NHWC;
            } else {
                TS_LOG_ERROR << this->op() << " do not support format: " << format << eject;
            }

            m_input_num = tensor::to_int(get("input_count"));
            m_out_num = tensor::to_int(get("output_count"));

            // m_model_path = "/sdcard/" + tensor::to_string(get(name::model_path));
            auto rknn_buffer = get("rknn_buffer");
            auto rknn_buffer_size = rknn_buffer.count();
            auto rknn_buffer_data = rknn_buffer.data<char>();

            int ret = 0;
            if (device_found) {
                // rknn_init_extend init_extend;
                // init_extend.device_id = device_name;
                // ret = rknn_init2(&m_ctx, rknn_buffer_data, rknn_buffer_size, RKNN_FLAG_PRIOR_MEDIUM, &init_extend);
                // if(ret < 0){
                // 	TS_LOG_ERROR << this->op() << " [RKNN] rknn_init2(\"" << device_name << "\") failed!"
                // 				 << " Error(" << ret << "): " << get_rknn_error_str(ret) << "." << eject;
                // }
            } else {
                ret = m_rknn->init(&m_ctx, rknn_buffer_data, rknn_buffer_size, RKNN_FLAG_PRIOR_MEDIUM, nullptr);
                if (ret < 0) {
                    TS_LOG_ERROR << this->op() << " [RKNN] rknn_init failed!"
                                 << " Error(" << ret << "): " << get_rknn_error_str(ret) << "." << eject;
                }
            }
            need incase_early_return(m_rknn->destroy, m_ctx);

            TS_LOG_INFO << this->op() << " [RKNN] rknn_init " << m_ctx << " succeed!";

            rknn_sdk_version sdk_version;
            ret = m_rknn->query(m_ctx, RKNN_QUERY_SDK_VERSION, &sdk_version, sizeof(sdk_version));
            if(ret < 0){
                TS_LOG_ERROR << this->op() << " [RKNN] sdk_version failed!"
                             << " Error(" << ret << "): " << get_rknn_error_str(ret) << "." << eject;
            }
            TS_LOG_INFO << this->op() << " [RKNN] api_version:" << sdk_version.api_version
                        << ", driver_version:" << sdk_version.drv_version;

            rknn_input_output_num in_out_num;
            ret = m_rknn->query(m_ctx, RKNN_QUERY_IN_OUT_NUM, &in_out_num, sizeof(in_out_num));
            if (ret < 0) {
                TS_LOG_ERROR << this->op() << " [RKNN] rknn_query RKNN_QUERY_IN_OUT_NUM failed!"
                             << " Error(" << ret << "): " << get_rknn_error_str(ret) << "." << eject;
            }

            //check input and output num
            if (m_input_num != in_out_num.n_input) {
                TS_LOG_ERROR << this->op() << " input num not equale to model's input num,input num is "
                             << m_input_num << " but model's input num is " << in_out_num.n_input << "." << eject;
            }
            if (m_out_num != in_out_num.n_output) {
                TS_LOG_ERROR << this->op() << " output num not equale to model's output num,ouput num is "
                             << m_out_num << " but model's output num is " << in_out_num.n_output << "." << eject;
            }

            m_in_attrs.resize(m_input_num);
            m_out_attrs.resize(m_out_num);
            for (int i = 0; i < m_input_num; ++i) {
                m_in_attrs[i].index = i;
            }
            for (int i = 0; i < m_out_num; ++i) {
                m_out_attrs[i].index = i;
            }

            for (int i = 0; i < m_input_num; ++i) {
                ret = m_rknn->query(m_ctx, RKNN_QUERY_INPUT_ATTR, &(m_in_attrs[i]), sizeof(m_in_attrs[i]));
                if (ret < 0) {
                    TS_LOG_ERROR << this->op() << " [RKNN] query rknn_query RKNN_QUERY_INPUT_ATTR " << i << " failed!"
                                 << " Error(" << ret << "): " << get_rknn_error_str(ret) << "." << eject;
                }
                {
                    TS_LOG_DEBUG << "Input " << i << " format(" << m_in_attrs[i].fmt << "): "
                                 << shape_string(m_in_attrs[i].dims, m_in_attrs[i].n_dims)
                                 << " stride[h, w]=["
                                 << m_in_attrs[i].h_stride << ", " << m_in_attrs[i].w_stride << "]"
                                 << " type=" << type_str(get_ts_type_from_rknn(m_in_attrs[i].type));
                }
            }
            for (int i = 0; i < m_out_num; ++i) {
                ret = m_rknn->query(m_ctx, RKNN_QUERY_OUTPUT_ATTR, &(m_out_attrs[i]), sizeof(m_out_attrs[i]));
                if (ret < 0) {
                    TS_LOG_ERROR << this->op() << " [RKNN] query rknn_query RKNN_QUERY_OUTPUT_ATTR " << i << " failed!"
                                 << " Error(" << ret << "): " << get_rknn_error_str(ret) << "." << eject;
                }
                {
                    TS_LOG_DEBUG << "Output " << i << " format(" << m_out_attrs[i].fmt << "): "
                                 << shape_string(m_out_attrs[i].dims, m_out_attrs[i].n_dims)
                                 << " stride[h, w]=["
                                 << m_out_attrs[i].h_stride << ", " << m_out_attrs[i].w_stride << "]"
                                 << " type=" << type_str(get_ts_type_from_rknn(m_out_attrs[i].type));
                }
            }

            //check format
            // for (int i = 0; i < m_input_num; ++i) {
            //     if(m_data_format == DataFormat::DATA_NCHW){
            //         if(m_in_attrs[i].fmt != RKNN_TENSOR_NCHW){
            //             TS_LOG_ERROR << "input data format dismatch,current data format is not nchw" << eject;
            //         }
            //     }
            //     else{
            //         if(m_in_attrs[i].fmt != RKNN_TENSOR_NHWC){
            //             TS_LOG_ERROR << "input data format dismatch,current data format is not nhwc" << eject;
            //         }
            //     }
            // }

            for (int i = 0; i < m_out_num; ++i) {
                if (m_data_format == DataFormat::DATA_NCHW) {
                    if (m_out_attrs[i].fmt != RKNN_TENSOR_NCHW) {
                        TS_LOG_ERROR << this->op()
                                     << " output data format dismatch, query rknn data format is not nchw." << eject;
                    }
                } else {
                    if (m_out_attrs[i].fmt != RKNN_TENSOR_NHWC) {
                        TS_LOG_ERROR << this->op()
                                     << " output data format dismatch, query rknn data format is not nhwc." << eject;
                    }
                }
            }

            for (int i = 0; i < m_input_num; ++i) {
                auto &attr = m_in_attrs[i];
                uint32_t h, w;
                if (attr.fmt == RKNN_TENSOR_NHWC) {
                    h = attr.dims[1];
                    w = attr.dims[2];
                } else if (attr.fmt == RKNN_TENSOR_NCHW) {
                    h = attr.dims[2];
                    w = attr.dims[3];
                } else {
                    continue;
                }
                auto h_stride = attr.h_stride;
                auto w_stride = attr.w_stride;
                if ((h_stride != 0 && h_stride != h) || (w_stride != 0 && w_stride != w)) {
                    TS_LOG_ERROR << "Unsupported input(" << i << ") format("
                                 << get_rknn_format_string(attr.fmt) << "): "
                                 << shape_string(attr.dims, attr.n_dims)
                                 << " stride[h, w]=["
                                 << attr.h_stride << ", " << attr.w_stride << "]"
                                 << " type=" << type_str(get_ts_type_from_rknn(attr.type));
                }
            }

            for (int i = 0; i < m_out_num; ++i) {
                auto &attr = m_out_attrs[i];
                uint32_t h, w;
                if (attr.fmt == RKNN_TENSOR_NHWC) {
                    h = attr.dims[1];
                    w = attr.dims[2];
                } else if (attr.fmt == RKNN_TENSOR_NCHW) {
                    h = attr.dims[2];
                    w = attr.dims[3];
                } else {
                    continue;
                }
                auto h_stride = attr.h_stride;
                auto w_stride = attr.w_stride;
                if ((h_stride != 0 && h_stride != h) || (w_stride != 0 && w_stride != w)) {
                    TS_LOG_ERROR << "Unsupported output(" << i << ") format("
                                 << get_rknn_format_string(attr.fmt) << "): "
                                 << shape_string(attr.dims, attr.n_dims)
                                 << " stride[h, w]=["
                                 << attr.h_stride << ", " << attr.w_stride << "]"
                                 << " type=" << type_str(get_ts_type_from_rknn(attr.type));
                }
            }

            incase_early_return.release();  // no need to care
        }

        int Rknn2::infer(Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
            for (int i = 0; i < m_out_num; ++i) {
                Tensor::Prototype out_proto;
                // DTYPE out_type = get_ts_type_from_rknn(m_out_attrs[i].type);
                //NOTE:The output is float[default]
                DTYPE out_type = FLOAT32;
                int dims = (int)m_out_attrs[i].n_dims;
                Shape out_shape(dims);
                for (decltype(dims) j = 0; j < dims; ++j) {
                    out_shape[j] = (int)m_out_attrs[i].dims[j];
                }
                out_proto = Tensor::Prototype(out_type, out_shape);
                output.emplace_back(out_proto);
            }

            return 1;
        }

        Operator::shared Rknn2::_get_op_nchw2nhwc() {
            if (m_nchw2nhwc) return m_nchw2nhwc;
            auto transpose = OperatorCreator::Create(
                    "cpu", name::layer::transpose(), false);
            transpose->set(Bubble::RetentionParam::op, tensor::from(name::layer::transpose()));
            transpose->set(Bubble::RetentionParam::name, tensor::from(name() + "__nchw2nhwc"));
            transpose->set(name::permute, tensor::build(
                    INT32, std::vector<int32_t>({0, 2, 3, 1})));
            return transpose;
        }
        Operator::shared Rknn2::_get_op_nhwc2nchw() {
            if (m_nhwc2nchw) return m_nhwc2nchw;
            auto transpose = OperatorCreator::Create(
                    "cpu", name::layer::transpose(), false);
            transpose->set(Bubble::RetentionParam::op, tensor::from(name::layer::transpose()));
            transpose->set(Bubble::RetentionParam::name, tensor::from(name() + "__nhwc2nchw"));
            transpose->set(name::permute, tensor::build(
                    INT32, std::vector<int32_t>({0, 3, 1, 2})));
            return transpose;
        }

        static bool check_dims(const int *ts_data, size_t ts_size,
                               const uint32_t *rknn_data, uint32_t rknn_size) {
            if (ts_size != rknn_size) return false;
            for (decltype(rknn_size) i = 0; i < rknn_size; ++i) {
                if (ts_data[i] != rknn_data[i]) return false;
            }
            return true;
        }

        int Rknn2::run(Stack &stack) {
            std::vector<Tensor::Prototype> output_protos;
            infer(stack, output_protos);

            auto memory_device = running_memory_device();

            std::vector<Tensor> inputs;
            for (int i = 0; i < m_input_num; ++i) {
                inputs.emplace_back(stack[i].view(memory_device));
            }

            // check shape and NCHW & NHWC
            for (int i = 0; i < m_input_num; ++i) {
                // check format
                const auto got_fmt = m_data_format == DATA_NCHW ? RKNN_TENSOR_NCHW : RKNN_TENSOR_NHWC;
                if (m_in_attrs[i].fmt != got_fmt) {
                    if (m_in_attrs[i].fmt == RKNN_TENSOR_NCHW) {
                        inputs[i] = intime::transpose(inputs[i], {0, 3, 1, 2});
                    } else if (m_in_attrs[i].fmt == RKNN_TENSOR_NHWC) {
                        inputs[i] = intime::transpose(inputs[i], {0, 2, 3, 1});
                    } else {
                        TS_LOG_ERROR << op() << " Can not convert rknn input(" << i << ") from "
                                     << (m_data_format == DATA_NCHW ? "NCHW" : "NHWC") << " to fmt "
                                     << get_rknn_format_string(m_in_attrs[i].fmt) << eject;
                    }
                }
                // check shape
                if (!check_dims(inputs[i].sizes().data(), inputs[i].dims(),
                                m_in_attrs[i].dims, m_in_attrs[i].n_dims)) {
                    TS_LOG_ERROR << op() << " Input(" << i << ") mismatch from ts " << inputs[i].proto()
                                 << " to rknn "
                                 << shape_string(m_in_attrs[i].dims, m_in_attrs[i].n_dims)
                                 << eject;
                }
            }

            std::vector<Tensor> outputs(m_out_num);
            for (int i = 0; i < m_out_num; ++i) {
                outputs[i] = stack.make(output_protos[i]);
//                outputs[i] = Tensor(Tensor::InFlow::HOST, output_protos[i]);
            }

            rknn_forward(inputs, m_ctx, m_out_num, m_data_format, m_in_attrs, m_out_attrs, outputs);

            Tensor output;
            if (m_out_num > 1) {
                output.pack(outputs);
            } else {
                output = outputs[0];
            }
            stack.push(output);

            return 1;
        }

    } //base

} //ts
