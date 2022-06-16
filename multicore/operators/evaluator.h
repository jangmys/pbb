#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "../../common/include/subproblem.h"
#include "libbounds.h"

#include <memory>

template<typename T>
class Evaluator
{
public:
    Evaluator(){};
    Evaluator(std::unique_ptr<bound_abstract<T>> _lb) : lb(std::move(_lb)){};
    Evaluator(std::unique_ptr<bound_abstract<T>> _lb,
              std::unique_ptr<bound_abstract<T>> _lb2
    ) : lb(std::move(_lb)),
        lb2(std::move(_lb2))
    {};

    enum lb_strength{
        Primary = 0,
        Secondary = 1
    };

    void get_children_bounds_full(subproblem& node,std::vector<bool> mask, int fillPos, std::vector<T>& lower_bounds, std::vector<T>& priority,T best = std::numeric_limits<T>::max(),lb_strength lb_type=Primary);

    // void get_children_bounds_incr(subproblem& node, std::vector<T>& lower_bounds_begin, std::vector<T>& lower_bounds_end, std::vector<T>& priority_begin, std::vector<T>& priority_end, const int begin_end);
    //
    // void get_children_bounds_incr(subproblem& node, std::vector<T>& lower_bounds, std::vector<T>& priority, const int begin_end);

    T get_solution_cost(subproblem& s);


    bound_abstract<T>* get_lb(lb_strength);

private:
    std::unique_ptr<bound_abstract<T>> lb;
    std::unique_ptr<bound_abstract<T>> lb2;
    bool m_bidirectional = true;
};

#endif
