#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "../../common/include/subproblem.h"
#include "libbounds.h"

#include <memory>

//get children bounds BEGIN
//get children bounds END_ORDER
//get children bounds BEGIN_END

template<typename T>
class evaluator
{
public:
    evaluator(){};
    evaluator(std::unique_ptr<bound_abstract<T>> _lb) : lb(std::move(_lb)){};
    evaluator(std::unique_ptr<bound_abstract<T>> _lb,
              std::unique_ptr<bound_abstract<T>> _lb2
    ) : lb(std::move(_lb)),
        lb2(std::move(_lb2))
    {};

    std::unique_ptr<bound_abstract<T>> lb;
    std::unique_ptr<bound_abstract<T>> lb2;

    void get_children_bounds_strong(subproblem& node,std::vector<int> mask,int begin_end, int fillPos, std::vector<T>& lower_bounds, std::vector<T>& priority,int best = std::numeric_limits<T>::max());

    void get_children_bounds_weak(subproblem& node, std::vector<T>& lower_bounds_begin, std::vector<T>& lower_bounds_end, std::vector<T>& priority_begin, std::vector<T>& priority_end);

    void getChildrenBounds(subproblem& s, unsigned int fixposition, std::vector<T>& LB, T best_cost = std::numeric_limits<T>::max());

    T getBounds(subproblem* s, int fixposition);

    T getSolutionCost(subproblem& s);
};

#endif
