//
// Created by kier on 2018/6/28.
//

#ifndef TENSORSTACK_RUNTIME_INSTRUCTION_H
#define TENSORSTACK_RUNTIME_INSTRUCTION_H

#include <memory>
#include "operator.h"

namespace ts {

    class Workbench;

    class Instruction {
    public:
        using self = Instruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        virtual ~Instruction() = default;

        virtual void run(Workbench &workbench) = 0;

        virtual std::string str() const;

        virtual std::string repr() const;
    };

    class LambdaInstruction : public Instruction {
    public:
        using self = LambdaInstruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = Instruction;

        using Lambda = std::function<void(Workbench &workbench)>;

        LambdaInstruction(const Lambda &lambda);

        LambdaInstruction(const Lambda &lambda, const std::string &description);

        void run(Workbench &workbench) final;

        std::string str() const final;

    private:
        Lambda m_lambda;
        std::string m_description;
    };

    class StackInstruction : public Instruction {
    public:
        using self = StackInstruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = Instruction;

        void run(Workbench &workbench) final;

        virtual void run(Stack &stack) = 0;
    };

    /**
     * \brief push data sagment to stack
     */
    class DataSagmentInstruction : public Instruction {
    public:
        using self = DataSagmentInstruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = Instruction;

        explicit DataSagmentInstruction(int data_index);

        int data_index() const { return m_data_index; }

        void run(Workbench &workbench) final;

        std::string str() const final;

    private:
        int m_data_index;
    };

    class OperatorInstruction : public Instruction {
    public:
        using self = OperatorInstruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = Instruction;

        explicit OperatorInstruction(const Operator::shared &func, int nargs, int nresults);
        explicit OperatorInstruction(const Operator::shared &func, int nargs, int nresults, const std::string &description);

        void run(Workbench &workbench) final ;

        std::string str() const final;

    private:
        Operator::shared m_func = nullptr;
        int m_nargs = 0;
        int m_nresults = 0;
        std::string m_description;
    };

}


#endif //TENSORSTACK_RUNTIME_INSTRUCTION_H
