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

namespace ts {
    Workbench::Workbench(const ComputingDevice &device) {
        this->m_device_context.initialize(device);
        auto &memory_device = this->m_device_context.memory_device;

        this->m_static_memory = DynamicSyncMemoryController::Make(memory_device, true);
        // TODO: Make real flow memory controller
        this->m_flow_memory = HypeSyncMemoryController<FlowMemoryController>::Make(memory_device, false);
        this->m_dynamic_memory = DynamicSyncMemoryController::Make(memory_device, false);
        this->m_stack = std::make_shared<Stack>(memory_device, this->m_flow_memory);
        this->m_data_sagment = std::make_shared<Stack>(memory_device, this->m_static_memory);
    }

    Workbench::Workbench(const ComputingDevice &device, int computing_thread_number)
            : self(device) {
        this->m_runtime_context.set_computing_thread_number(computing_thread_number);
    }

    Workbench::~Workbench() {
        this->m_program.clear();
        this->m_stack->clear();
        this->m_map_input_slots.clear();
        this->m_map_output_slots.clear();
        this->m_inputs.clear();
        this->m_input_filters.clear();
        this->m_outputs.clear();
        this->m_device_context.finalize();
    }

    static Tensor filter_images(ImageFilter::shared filter, Tensor input) {
        if (input.dims() < 2 || input.dims() > 4) {
            TS_LOG_ERROR << "Can not filter input with shape: " << to_string(input.sizes()) << eject;
        }
        if (input.dims() == 2) {
            input = input.reshape({input.size(0), input.size(1), 1});
        }
        if (input.dims() == 3) {
            return filter->run(input);
        }
        if (input.size(0) == 1) {
            input = input.reshape({input.size(1), input.size(2), input.size(3)});
            auto result = filter->run(input);
            result = result.reshape({1, input.size(0), input.size(1), input.size(2)});
            return result;
        }
        // split batch and do filter
        auto number = input.size(0);
        auto left_size = input.sizes();
        left_size.erase(left_size.begin());
        auto left_count = 1;
        for (auto &size : left_size) left_count *= size;
        std::vector<Tensor> fields(number);
        for (int i = 0; i < number; ++i) {
            auto image_data = input.data<char>() + i * left_count * input.proto().type_bytes();
            Tensor image(Memory(input.device(), image_data, left_count),
                    Tensor::Prototype(input.dtype(), left_size));
            fields[i] = filter->run(image);
            // check shape
            if (i == 0) continue;
            if (!fields[i].has_shape(fields[i - 1].sizes())) {
                TS_LOG_ERROR << "Can not concat image " << to_string(fields[i].sizes())
                    << " vs. " << to_string(fields[i - 1].sizes());
            }
        }
        auto new_size = fields[0].sizes();
        new_size.insert(new_size.begin(), number);

        Tensor result(input.device(), input.dtype(), new_size);
        auto converter = HardConverter::Query(result.device().type(), input.device().type());
        TS_AUTO_CHECK(converter != nullptr);

        auto field_size = fields[0].proto().count() * fields[0].proto().type_bytes();

        for (int i = 0; i < fields.size(); ++i) {
            auto &field = fields[i];
            auto result_data = result.data<char>() + i * field_size;
            auto filed_data = result.data();
            converter(result.device().id(), result_data, field.device().id(), filed_data, field_size);
        }

        return result;
    }

    void Workbench::run() {
        // clear pre output
        this->m_pointer = 0;
        this->m_stack->rebase(0);
        this->m_stack->clear();
        this->m_outputs.resize(this->m_outputs.size(), Tensor());

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
        }

        // bind thread pool to any operator can using thread speed up
        ctx::bind<ThreadPool> bind_thread_pool(this->runtime().thread_pool());

        // bind device context
        ctx::bind<DeviceContext> bind_device_context(m_device_context);

        // bind runtime context
        ctx::bind<RuntimeContext> bind_runtime_context(m_runtime_context);

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
            this->m_outputs[i] = arg.clone(m_dynamic_memory, arg.device());
        }
    }

    Workbench::shared Workbench::clone() const {
        Workbench::shared dolly = std::make_shared<Workbench>(this->m_device_context.computing_device);
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

        return dolly;
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

        // never swallow any exception
        block = compiler.compile(module_inputs, module_outputs);

        // link data sagment
        // TODO: link multi-data-sagment
        auto data_sagment_base = int(bench->m_data_sagment->size());
        for (auto &inst : block.instructions) {
            auto data_sagment_inst = dynamic_cast<DataSagmentInstruction *>(inst.get());
            if (data_sagment_inst == nullptr) continue;
            inst = std::make_shared<DataSagmentInstruction>(data_sagment_inst->data_index() + data_sagment_base);
        }
        for (auto &data : block.data_sagment) {
            if (data.device.empty()) {
                bench->m_data_sagment->clone_push(data.tensor);
            } else {
                bench->m_data_sagment->clone_push(data.tensor, data.device);
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
        this->bind_filter(slot_it->second, filter);
    }

    void Workbench::bind_filter(int slot, ImageFilter::shared filter) {
        if (slot < 0 || size_t(slot) >= m_input_filters.size()) {
            TS_LOG_ERROR << "Input index out of range. with index=" << slot << eject;
        }
        m_input_filters[slot] = filter;
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
        this->m_stack->push(*this->m_data_sagment->index(data_index));
    }
}
