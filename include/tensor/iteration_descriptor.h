//
// Created by seeta on 2018/6/3.
//

#ifndef TENSORSTACK_TENSOR_ITERATION_DESCRIPTOR_H
#define TENSORSTACK_TENSOR_ITERATION_DESCRIPTOR_H

#include <vector>
#include <cstdint>
#include <climits>
#include <memory>
#include <iostream>

namespace ts {
    class Iteration {
    public:
        using step_t = long;    ///< step type
        static const step_t outer = LONG_MAX;   ///< means step into outer area
        static const step_t finish = LONG_MIN;  ///< means iteration was over
        using times_t = size_t;
        // using count_t = times_t; // no counter in this version

        enum Type {
            STEP,
            ITERS
        };

        explicit Iteration(Type type) : type(type) { std::cout << "+"; }

        Iteration(Type type, times_t times) : type(type), times(times) { std::cout << "+"; }

        virtual ~Iteration() { std::cout << "-"; };

        Type type;
        times_t times = 0;
        // count_t count = 0;
    };

    template<Iteration::Type _type>
    class IterationBase : public Iteration {
    public:
        IterationBase() : Iteration(_type) {}

        IterationBase(times_t times) : Iteration(_type, times) {}
    };

    class IterationStep : public IterationBase<Iteration::STEP> {
    public:
        using supper = IterationBase<Iteration::STEP>;

        IterationStep() = default;

        explicit IterationStep(times_t times) : supper(times) {}

        explicit IterationStep(times_t times, step_t step) : supper(times), step(step) {}

        step_t step = 0;
    };

    class IterationIters : public IterationBase<Iteration::ITERS> {
    public:
        using supper = IterationBase<Iteration::ITERS>;

        IterationIters() = default;

        explicit IterationIters(times_t times) : supper(times) {}

        explicit IterationIters(times_t times, const std::vector<Iteration *> &iters) : supper(times), iters(iters) {}

        explicit IterationIters(times_t times, std::vector<Iteration *> &&iters) : supper(times),
                                                                                   iters(std::move(iters)) {}

        std::vector<Iteration *> iters; // those pointers should allocate by new
    };

    inline IterationStep *new_iteration_step() {
        return new IterationStep();
    }

    inline IterationIters *new_iteration_iters() {
        return new IterationIters();
    }

    inline IterationStep *new_iteration_step(Iteration::times_t times) {
        return new IterationStep(times);
    }

    inline IterationIters *new_iteration_iters(Iteration::times_t times) {
        return new IterationIters(times);
    }

    inline Iteration *new_iteration(Iteration::times_t times, Iteration::step_t step) {
        return new IterationStep(times, step);
    }

    inline Iteration *new_iteration(Iteration::times_t times, Iteration *iters) {
        return new IterationIters(times, {iters});
    }

    inline Iteration *new_iteration(Iteration::times_t times, const std::vector<Iteration *> &iters) {
        return new IterationIters(times, iters);
    }

    inline Iteration *new_iteration(Iteration::times_t times, std::vector<Iteration *> &&iters) {
        return new IterationIters(times, std::move(iters));
    }

    inline void delete_iteration(const Iteration *iteration) {
        if (iteration == nullptr) return;
        if (iteration->type == Iteration::ITERS) {
            auto iters = reinterpret_cast<const IterationIters *>(iteration)->iters;
            for (auto *iter : iters) delete_iteration(iter);
        }
        delete iteration;
    }

    inline Iteration *clone_iteration(const Iteration *iteration) {
        if (iteration == nullptr) return nullptr;
        if (iteration->type == Iteration::ITERS) {
            using ThisIteration = IterationIters;
            auto iteration_iters = reinterpret_cast<const ThisIteration *>(iteration);
            std::unique_ptr<IterationIters, decltype(&delete_iteration)> iteration_clone(new_iteration_iters(),
                                                                                         &delete_iteration);
            iteration_clone->times = iteration_iters->times;
            iteration_clone->iters.resize(iteration_iters->iters.size(), nullptr);
            auto from = iteration_iters->iters.begin();
            auto to = iteration_clone->iters.begin();
            while (from != iteration_iters->iters.end()) {
                *to = clone_iteration(*from);
                ++from;
                ++to;
            }
            return iteration_clone.release();
        } else {
            using ThisIteration = IterationStep;
            auto iteration_step = reinterpret_cast<const ThisIteration *>(iteration);
            return new_iteration(iteration_step->times, iteration_step->step);
        }
    }

    class IterationDescrptor {
    public:
        using self = IterationDescrptor;

        IterationDescrptor(Iteration::times_t times, Iteration::step_t step) : base(new_iteration(times, step)) {}

        IterationDescrptor(Iteration::times_t times, const IterationDescrptor &descrptor);

        IterationDescrptor(Iteration::times_t times, IterationDescrptor &&descrptor);

        IterationDescrptor(Iteration::times_t times, std::initializer_list<IterationDescrptor> list);

        ~IterationDescrptor();

        self &operator=(const IterationDescrptor &descrptor);

        self &operator=(IterationDescrptor &&descrptor);

    private:
        IterationDescrptor(const IterationDescrptor &descrptor);

        IterationDescrptor(IterationDescrptor &&descrptor);

        Iteration *base = nullptr;
    };

}


#endif //TENSORSTACK_TENSOR_ITERATION_DESCRIPTOR_H
