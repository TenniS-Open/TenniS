//
// Created by kier on 2018/5/25.
//

#include <module/module.h>
#include <core/device.h>
#include <runtime/workbench.h>
#include <utils/ctxmgr.h>

#include "runtime/workbench.h"
#include "global/memory_device.h"
#include "compiler/compiler.h"
#include "utils/box.h"

#include "memory/flow.h"
#include "global/hard_converter.h"

#include <climits>
#include "utils/need.h"

#include "kernels/common/math.h"
#include "core/tensor_builder.h"
#include "backend/name.h"
#include "backend/base/base_cast_v2.h"
#include "runtime/operator.h"

#include "utils/ctxmgr_lite_support.h"

namespace ts {
    class BindWorkbenchRuntime {
    public:
        using self = BindWorkbenchRuntime;

        explicit BindWorkbenchRuntime(Workbench &bench)
            : bind_thread_pool(bench.runtime().thread_pool())
            , bind_device_context(bench.device())
            , bind_runtime_context(bench.runtime()) {
            bench.device().active();
            m_pre_device_context = DeviceContext::Switch(&bench.device());
        }

        ~BindWorkbenchRuntime() {
            DeviceContext::Switch(m_pre_device_context);
        }

    private:
        // bind thread pool to any operator can using thread speed up
        ctx::bind<ThreadPool> bind_thread_pool;

        // bind device context
        ctx::bind<DeviceContext> bind_device_context;

        // bind runtime context
        ctx::bind<RuntimeContext> bind_runtime_context;

        // pre_device_context
        DeviceContext *m_pre_device_context = nullptr;
    };

    Workbench::Workbench(const ComputingDevice &device, std::shared_ptr<std::mutex> mutex) {
        this->m_device_context.initialize(device);
        auto &memory_device = this->m_device_context.memory_device;

        this->m_static_memory = DynamicSyncMemoryController::Make(memory_device, true);
        // TODO: Make real flow memory controller
        this->m_flow_memory = HypeSyncMemoryController<FlowMemoryController>::Make(memory_device, false);
        this->m_dynamic_memory = DynamicSyncMemoryController::Make(memory_device, false);
        this->m_stack = std::make_shared<Stack>(memory_device, this->m_flow_memory);
        this->m_data_sagment = std::make_shared<Stack>(memory_device, this->m_static_memory);
        this->m_mutex = std::move(mutex);
        // bind flow and dynamic memory, so you can use it to alloc memory in any where
        this->m_runtime_context.bind_flow(this->m_flow_memory);
        this->m_runtime_context.bind_dynamic(this->m_dynamic_memory);
    }

    Workbench::Workbench(const ComputingDevice &device, std::shared_ptr<std::mutex> mutex, int computing_thread_number)
            : self(device, std::move(mutex)) {
        this->m_runtime_context.set_computing_thread_number(computing_thread_number);
    }

    Workbench::Workbench(const ComputingDevice &device)
            : self(device, std::make_shared<std::mutex>()) {
    }

    Workbench::Workbench(const ComputingDevice &device, int computing_thread_number)
            : self(device, std::make_shared<std::mutex>(), computing_thread_number) {
    }

    Workbench::~Workbench() {
        this->m_program.clear();
        this->m_stack->clear();
        this->m_map_input_slots.clear();
        this->m_map_output_slots.clear();
        this->m_inputs.clear();
        this->m_outputs.clear();
        this->m_input_filters.clear();
        this->m_device_context.finalize();
    }

    static Tensor filter_images(ImageFilter::shared filter, Tensor input) {
        if (input.dims() < 2 || input.dims() > 4) {
            TS_LOG_ERROR << "Can not filter input with shape: " << to_string(input.sizes()) << eject;
        }
        if (input.dims() == 2) {
            input = input.reshape({1, input.size(0), input.size(1), 1});
            auto output = filter->run(input);
            return output;
        }
        if (input.dims() == 3) {
            input = input.reshape({1, input.size(0), input.size(1), input.size(2)});
            auto output = filter->run(input);
            return output;
        }
        return filter->run(input);
    }

    static inline std::unique_ptr<ctx::bind<Profiler>> bind_profiler(bool _do, Profiler &profiler) {
        if (!_do) return nullptr;
        return std::unique_ptr<ctx::bind<Profiler>>(new ctx::bind<Profiler>(profiler));
    }

    void Workbench::run() {
        // clear pre output
        this->m_pointer = 0;
        this->m_stack->rebase(0);
        this->m_stack->clear();
        this->m_outputs.resize(this->m_outputs.size(), Tensor());

        BindWorkbenchRuntime _bind_runtime(*this);

        // set input
        for (int i = 0; static_cast<size_t >(i) < this->m_inputs.size(); ++i) {
            auto &arg = this->m_inputs[i];
            auto &filter = this->m_input_filters[i];
            if (filter == nullptr) {
                this->m_stack->push(arg);
            } else {
                // TODO: do filter
                this->m_stack->push(filter_images(filter, arg));
            }
            auto dtype = m_input_dtypes[i];
            if (dtype == VOID) continue;
            cast_tensor(dtype);
        }

        auto _bind_profiler = bind_profiler(m_do_profile, m_profiler);

        // run
        while (m_pointer < m_program.size()) {
            auto &inst = m_program[m_pointer];
            m_pointer++;
            inst->run(*this);
        }

        // check output
        if (this->m_stack->size() != this->m_outputs.size()) {
            TS_LOG_ERROR << "Got unexpected output number want " << this->m_outputs.size() << " vs. "
                         << this->m_stack->size() << " given." << eject;
        }

        // set output
        // TODO: change output memory device type
        for (int i = 0; static_cast<size_t >(i) < this->m_stack->size(); ++i) {
            auto &arg = *this->m_stack->index(i);
            // this->m_outputs[i] = arg.clone(m_dynamic_memory, arg.device());
            this->m_outputs[i] = arg;   // do not clone by default
        }
    }

    Workbench::shared Workbench::clone() const {
        std::unique_lock<std::mutex> _lock_clone(*this->m_mutex);

        Workbench::shared dolly(new Workbench(
                this->m_device_context.computing_device,
                this->m_mutex));
        dolly->m_pointer = this->m_pointer;
        dolly->m_program = this->m_program;
        dolly->m_inputs.resize(this->m_inputs.size());
        dolly->m_outputs.resize(this->m_outputs.size());
        dolly->m_map_input_slots = this->m_map_input_slots;
        dolly->m_map_output_slots = this->m_map_output_slots;
        dolly->m_data_sagment = this->m_data_sagment;
        dolly->m_runtime_context = this->m_runtime_context.clone();
        dolly->m_input_filters.resize(this->m_input_filters.size(), nullptr);

        for (size_t i = 0; i < this->m_input_filters.size(); ++i) {
            if (this->m_input_filters[i] == nullptr) continue;
            dolly->m_input_filters[i] = this->m_input_filters[i]->clone();
        }

        // bind device context
        ctx::bind<DeviceContext> bind_device_context(const_cast<DeviceContext*>(&this->m_device_context));

        for (auto &instruction : dolly->m_program) {
            auto op = dynamic_cast<OperatorInstruction*>(instruction.get());
            if (op == nullptr) continue;
            instruction = op->clone();
        }

        // copy dtype
        dolly->m_input_dtypes = m_input_dtypes;
        dolly->m_output_dtypes = m_output_dtypes;

        return std::move(dolly);
    }

    template <typename T>
    inline void filter_values(T *value, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            if (near(*value, T(0))) *value = T(0);
            ++value;
        }
    }

    template <>
    inline void filter_values(float *value, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            if (*value < FLT_EPSILON && -*value < FLT_EPSILON) *value = 0;
            ++value;
        }
    }

    template <>
    inline void filter_values(double *value, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            if (*value < DBL_EPSILON && -*value < DBL_EPSILON) *value = 0;
            ++value;
        }
    }

    Workbench::shared Workbench::Load(const Module::shared &module, const ComputingDevice &device) {
        auto bench = std::make_shared<Workbench>(device);
        // convert module to bench
        // here running compilation codes
        // TODO: support RNN
        Compiler compiler(device);
        auto module_inputs = module->inputs();
        auto module_outputs = module->outputs();
        InstructionBlock block;

        {
            // bind bench for intime action
            ctx::bind<Workbench> _bind_workbench(*bench);

            // never swallow any exception
            block = compiler.compile(module_inputs, module_outputs);
        }

        // link data sagment
        // TODO: link multi-data-sagment
        auto data_sagment_base = int(bench->m_data_sagment->size());
        for (auto &inst : block.instructions) {
            auto data_sagment_inst = dynamic_cast<DataSagmentInstruction *>(inst.get());
            if (data_sagment_inst == nullptr) continue;
            inst = std::make_shared<DataSagmentInstruction>(data_sagment_inst->data_index() + data_sagment_base);
        }
        for (auto &data : block.data_sagment) {
            Tensor *value = nullptr;
            if (data.device.empty()) {
                value = bench->m_data_sagment->clone_push(data.tensor);
            } else {
                value = bench->m_data_sagment->clone_push(data.tensor, data.device);
            }

            // filter value
            if (value->device().type() == CPU) {
                if (value->dtype() == FLOAT32) {
                    filter_values(value->data<float>(), size_t(value->count()));
                } else if (value->dtype() == FLOAT64) {
                    filter_values(value->data<double>(), size_t(value->count()));
                }
            }
        }

        // binding instructions
        bench->m_program = block.instructions;
        // binding input and output shots
        bench->m_inputs.resize(module_inputs.size());
        bench->m_input_filters.resize(module_inputs.size());
        bench->m_outputs.resize(module_outputs.size());
        int slot_i = 0;
        for (auto &input : module_inputs) {
            bench->m_map_input_slots.insert(std::make_pair(input.bubble().name(), slot_i++));
        }
        slot_i = 0;
        for (auto &output : module_outputs) {
            bench->m_map_output_slots.insert(std::make_pair(output.bubble().name(), slot_i++));
        }

        bench->m_input_dtypes.resize(module_inputs.size(), VOID);
        bench->m_output_dtypes.resize(module_outputs.size(), VOID);

        for (size_t i = 0; i < module_inputs.size(); ++i) {
            auto &input = module_inputs[i];
            if (!input->has(Bubble::RetentionParam::dtype)) continue;
            bench->m_input_dtypes[i] =
                    DTYPE(tensor::to_int(input->get(Bubble::RetentionParam::dtype)));
        }
        for (size_t i = 0; i < module_outputs.size(); ++i) {
            auto &output = module_outputs[i];
            if (!output->has(Bubble::RetentionParam::dtype)) continue;
            bench->m_output_dtypes[i] =
                    DTYPE(tensor::to_int(output->get(Bubble::RetentionParam::dtype)));
        }

        return bench;
    }

    void Workbench::jump_relative(int shift) { this->m_pointer += shift; }

    void Workbench::jump_absolute(size_t pointer) { this->m_pointer = pointer; }

    void Workbench::input(int slot, const Tensor &tensor) {
        if (slot < 0 || size_t(slot) >= m_inputs.size()) {
            TS_LOG_ERROR << "Input index out of range. with index=" << slot << eject;
        }
        m_inputs[slot] = tensor;
    }

    const Tensor &Workbench::output(int slot) const {
        if (slot < 0 || size_t(slot) >= m_outputs.size()) {
            TS_LOG_ERROR << "Output index out of range. with index=" << slot << eject;
        }
        return m_outputs[slot];
    }

    int Workbench::input_count() const {
        return int(m_inputs.size());
    }

    int Workbench::output_count() const {
        return int(m_outputs.size());
    }

    const Tensor &Workbench::input(int slot) const {
        if (slot < 0 || size_t(slot) >= m_inputs.size()) {
            TS_LOG_ERROR << "Input index out of range. with index=" << slot << eject;
        }
        return m_inputs[slot];
    }

    static std::string fuzzy_name(const Workbench::map<std::string, int> &map_name_slot, const std::string &name) {
        if (map_name_slot.empty()) return "";
        int min_edit_distance = INT_MAX;
        std::string closest_name;
        for (auto &name_slot_pair : map_name_slot) {
            auto &target_name = name_slot_pair.first;
            int dist = edit_distance(name, target_name);
            if (dist < min_edit_distance) {
                closest_name = target_name;
                min_edit_distance = dist;
            }
        }
        return closest_name;
    }

    void Workbench::bind_filter(const std::string &name, ImageFilter::shared filter) {
        auto slot_it = m_map_input_slots.find(name);
        if (slot_it == m_map_input_slots.end()) {
            TS_LOG_ERROR << "Can not identify the name \"" << name << "\", did you mean: "
                         << fuzzy_name(m_map_input_slots, name) << eject;
        }
        this->bind_filter(slot_it->second, std::move(filter));
    }

    void Workbench::bind_filter(int slot, ImageFilter::shared filter) {
        if (slot < 0 || size_t(slot) >= m_input_filters.size()) {
            TS_LOG_ERROR << "Input index out of range. with index=" << slot << eject;
        }
        filter->compile();
        m_input_filters[slot] = std::move(filter);
    }

    void Workbench::input(const std::string &name, const Tensor &tensor) {
        auto slot_it = m_map_input_slots.find(name);
        if (slot_it == m_map_input_slots.end()) {
            TS_LOG_ERROR << "Can not identify the name \"" << name << "\", did you mean: "
                         << fuzzy_name(m_map_input_slots, name) << eject;
        }
        this->input(slot_it->second, tensor);
    }

    const Tensor &Workbench::output(const std::string &name) const {
        auto slot_it = m_map_output_slots.find(name);
        if (slot_it == m_map_output_slots.end()) {
            TS_LOG_ERROR << "Can not identify the name \"" << name << "\", did you mean: "
                         << fuzzy_name(m_map_output_slots, name) << eject;
        }
        return this->output(slot_it->second);
    }

    const Tensor &Workbench::input(const std::string &name) const {
        auto slot_it = m_map_input_slots.find(name);
        if (slot_it == m_map_input_slots.end()) {
            TS_LOG_ERROR << "Can not identify the name \"" << name << "\", did you mean: "
                         << fuzzy_name(m_map_input_slots, name) << eject;
        }
        return this->input(slot_it->second);
    }

    void Workbench::push_data_sagment(int data_index) {
        // TODO: deal with data_sagment, in case of thread sharing, waitting testing
        this->m_stack->push(this->m_data_sagment->index(data_index)->weak());
    }

    Operator::shared Workbench::offline_create(const Bubble &bubble, bool strict) {
        // bind device context
        ctx::bind<DeviceContext> bind_device_context(m_device_context);

        auto built_op = OperatorCreator::Create(m_device_context.computing_device.type(), bubble.op(), strict);

        if (built_op == nullptr) return nullptr;

        for (auto &param : bubble.params()) {
            built_op->set(param.first, param.second);
        }

        built_op->init();

        return built_op;
    }

    void Workbench::offline_run(Operator::shared op, const std::vector<Tensor> &input, std::vector<Tensor> &output) {
        Stack stack(m_device_context.memory_device, m_dynamic_memory);

        BindWorkbenchRuntime _bind_runtime(*this);

        for (auto &tensor : input) {
            stack.push(tensor);
        }

        auto output_count = RunOperator(op, stack, int(input.size()));

        TS_AUTO_CHECK(output_count == stack.size());

        output.resize(output_count);

        for (int i = 0; i < output_count; ++i) {
            output[i] = stack[i];
        }
    }

    void Workbench::offline_infer(Operator::shared op, const std::vector<Tensor> &input,
                                  std::vector<Tensor::Prototype> &output) {
        Stack stack(m_device_context.memory_device, m_dynamic_memory);

        BindWorkbenchRuntime _bind_runtime(*this);

        for (auto &tensor : input) {
            stack.push(tensor);
        }

        op->infer(stack, output);
    }

    void Workbench::set_operator_param(const std::string &node_name, const std::string &param, const Tensor &value) {
        for (auto &inst : m_program) {
            auto* operator_inst = dynamic_cast<OperatorInstruction*>(inst.get());
            if (operator_inst == nullptr) continue;
            auto node = operator_inst->op();
            if (node->name() != node_name) continue;
            node->set(param, value);
            node->init();
        }
    }

    void Workbench::cast_tensor(DTYPE dtype) {
        if (m_cast_op == nullptr) {
            m_cast_op = OperatorCreator::Create(
                    m_device_context.computing_device.type(),
                    name::layer::cast(), false);
        }
        auto *cast_op = dynamic_cast<base::CastV2*>(m_cast_op.get());
        if (cast_op != nullptr) {
            cast_op->set_dtype(dtype);
        } else {
            m_cast_op->set(name::dtype, tensor::from<int32_t>(dtype));
            m_cast_op->init();
        }
        TS_AUTO_CHECK(1 == RunOperator(m_cast_op, *m_stack, 1));
    }

    Operator::shared Workbench::online_create(const Bubble &bubble, bool strict) {
        return offline_create(bubble, strict);
    }

    int Workbench::online_run(Operator::shared op, const std::vector<Tensor> &input) {
        m_stack->clear();
        for (auto &tensor : input) {
            m_stack->push(tensor);
        }
        return online_run(op, stack().size());
    }

    int Workbench::online_run(Operator::shared op, int argc) {
        BindWorkbenchRuntime _bind_runtime(*this);

        return RunOperator(op, *m_stack, argc);
    }

    void Workbench::online_run(Instruction::shared inst) {
        BindWorkbenchRuntime _bind_runtime(*this);

        inst->run(*this);
    }

    void Workbench::online_run(Instruction::shared inst, const std::vector<Tensor> &input) {
        m_stack->clear();
        for (auto &tensor : input) {
            m_stack->push(tensor);
        }
        online_run(inst);
    }

    int Workbench::online_run(const Bubble &bubble, int argc, bool strict) {
        return online_run(online_create(bubble, strict), argc);
    }

    int Workbench::online_run(const Bubble &bubble, const std::vector<Tensor> &input, bool strict) {
        return online_run(online_create(bubble, strict), input);
    }
}

TS_LITE_CONTEXT(ts::Workbench)
