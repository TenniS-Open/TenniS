//
// Created by seeta on 2018/5/25.
//

#include <module/module.h>
#include <core/device.h>
#include <runtime/workbench.h>

#include "runtime/workbench.h"
#include "global/memory_device.h"
#include "compiler/compiler.h"

namespace ts {
    Workbench::Workbench(const ComputingDevice &device)
            : m_device(device) {
        auto memory_device = QueryMemoryDevice(device);
        this->m_static_memory = std::make_shared<DynamicMemoryController>(memory_device);
        this->m_flow_memory = std::make_shared<DynamicMemoryController>(
                memory_device);   // TODO: Make real flow memory controller
        this->m_dynamic_memory = std::make_shared<DynamicMemoryController>(memory_device);
        this->m_stack = std::make_shared<Stack>(memory_device, this->m_flow_memory);
        this->m_data_sagment = std::make_shared<Stack>(memory_device, this->m_static_memory);
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
            this->m_stack->clone_push(arg);
        }

        // run
        while (m_pointer < m_program.size()) {
            auto &inst = m_program[m_pointer];
            m_pointer++;
            inst->run(*this);
        }

        // check output
        if (this->m_stack->size() != this->m_outputs.size()) {
            throw Exception("Got unexpected output number want " + std::to_string(this->m_outputs.size()) +
                            " vs. " + std::to_string(this->m_stack->size()) + " given.");
        }

        // set output
        // TODO: change output memory device type
        for (int i = 0; static_cast<size_t >(i) < this->m_stack->size(); ++i) {
            auto &arg = *this->m_stack->index(i);
            this->m_outputs[i] = arg.clone(m_dynamic_memory);
        }
    }

    Workbench::shared Workbench::clone() const {
        Workbench::shared dolly = std::make_shared<Workbench>(this->m_device);
        dolly->m_pointer = this->m_pointer;
        dolly->m_program = this->m_program;
        dolly->m_inputs.resize(this->m_inputs.size());
        dolly->m_outputs.resize(this->m_outputs.size());
        dolly->m_map_input_slots = this->m_map_input_slots;
        dolly->m_map_output_slots = this->m_map_output_slots;
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
        try {
            block = compiler.compile(module_inputs, module_outputs);
        } catch (const Exception &e) {
            return nullptr;
        }

        // binding instructions
        bench->m_program = block.instructions;
        // binding input and output shots
        bench->m_inputs.resize(module_inputs.size());
        bench->m_outputs.resize(module_outputs.size());
        int slot_i = 0;
        for (auto &input : module_inputs) {
            bench->m_map_input_slots.insert(std::make_pair(input.ref<Bubble>().name(), slot_i++));
        }
        slot_i = 0;
        for (auto &output : module_outputs) {
            bench->m_map_output_slots.insert(std::make_pair(output.ref<Bubble>().name(), slot_i++));
        }

        return bench;
    }

    void Workbench::jump_relative(int shift) { this->m_pointer += shift; }

    void Workbench::jump_absolute(size_t pointer) { this->m_pointer = pointer; }

    void Workbench::input(int slot, const Tensor &tensor) {
        if (slot < 0 || size_t(slot) >= m_inputs.size()) {
            throw Exception("Input index out of range. with index=" + std::to_string(slot));
        }
        m_inputs[slot] = tensor;
    }

    const Tensor &Workbench::output(int slot) const {
        if (slot < 0 || size_t(slot) >= m_outputs.size()) {
            throw Exception("Output index out of range, with index= " + std::to_string(slot));
        }
        return m_outputs[slot];
    }

    const Tensor &Workbench::input(int slot) const {
        if (slot < 0 || size_t(slot) >= m_inputs.size()) {
            throw Exception("Input index out of range. with index=" + std::to_string(slot));
        }
        return m_inputs[slot];
    }

    template<typename T>
    static inline T min(T a, T b, T c) {
        return std::min<T>(std::min<T>(a, b), c);
    }

    static inline int edit_distance(const std::string &lhs, const std::string &rhs) {
        const size_t M = lhs.length();  // rows
        const size_t N = rhs.length();  // cols

        if (M == 0) return int(N);
        if (N == 0) return int(M);

        std::unique_ptr<int[]> dist(new int[M * N]);
#define __EDIT_DIST(m, n) (dist[(m) * N + (n)])
        __EDIT_DIST(0, 0) = lhs[0] == rhs[0] ? 0 : 2;
        for (size_t n = 1; n < N; ++n) {
            __EDIT_DIST(0, n) = __EDIT_DIST(0, n - 1) + 1;
        }
        for (size_t m = 1; m < M; ++m) {
            __EDIT_DIST(m, 0) = __EDIT_DIST(m - 1, 0) + 1;
        }
        for (size_t m = 1; m < M; ++m) {
            for (size_t n = 1; n < N; ++n) {
                if (lhs[m] == rhs[n]) {
                    __EDIT_DIST(m, n) = min(
                            __EDIT_DIST(m - 1, n),
                            __EDIT_DIST(m, n - 1),
                            __EDIT_DIST(m - 1, n - 1));
                } else {
                    __EDIT_DIST(m, n) = min(
                            __EDIT_DIST(m - 1, n) + 1,
                            __EDIT_DIST(m, n - 1) + 1,
                            __EDIT_DIST(m - 1, n - 1) + 2);
                }
            }
        }
        return dist[M * N - 1];
#undef __EDIT_DIST
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

    void Workbench::input(const std::string &name, const Tensor &tensor) {
        auto slot_it = m_map_input_slots.find(name);
        if (slot_it == m_map_input_slots.end()) {
            throw Exception(
                    "Can not identify the name \"" + name + "\", did you mean: " + fuzzy_name(m_map_input_slots, name));
        }
        this->input(slot_it->second, tensor);
    }

    const Tensor &Workbench::output(const std::string &name) const {
        auto slot_it = m_map_output_slots.find(name);
        if (slot_it == m_map_output_slots.end()) {
            throw Exception("Can not identify the name \"" + name + "\", did you mean: " +
                            fuzzy_name(m_map_output_slots, name));
        }
        return this->output(slot_it->second);
    }

    const Tensor &Workbench::input(const std::string &name) const {
        auto slot_it = m_map_input_slots.find(name);
        if (slot_it == m_map_input_slots.end()) {
            throw Exception(
                    "Can not identify the name \"" + name + "\", did you mean: " + fuzzy_name(m_map_input_slots, name));
        }
        return this->input(slot_it->second);
    }

    void Workbench::push_data_sagment(int data_index) {
        this->m_stack->push(*this->m_data_sagment->index(data_index));
    }
}
