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

    std::unique_ptr<bound_abstract<T>> lb;

    void getChildrenBounds(subproblem& s, unsigned int fixposition, std::vector<T>& LB, T best_cost = std::numeric_limits<T>::max());

    T getBounds(subproblem* s, int fixposition);

    T getSolutionCost(subproblem& s);
};

#endif
