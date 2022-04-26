#include <limits>
#include <algorithm>

#include "branching.h"
#include "evaluator.h"


template<typename T>
bound_abstract<T>*
evaluator<T>::get_lb(evaluator<T>::lb_strength _lb)
{
    if(_lb == evaluator<T>::Primary){
        return lb.get();
    }else{
        return lb2.get();
    }
}

/**
 * Compute children's bounds using a single-child bounding function, either front or back
 *
 * @param node Subproblem (parent)
 * @param mask vector<bool> mask[i]==true iff bound for job i is computed
 * @param fillPos int position where unscheduled jobs are fixed
 * @param (out) lower_bounds vector - LB for job j at index j
 * @param (out) priority vector - priority value for child j at index j (if provided by bound)
 * @param best int best known UB (for early exit in bound)
*/
template<typename T>
void
evaluator<T>::get_children_bounds_full(subproblem& node, std::vector<bool> mask, int fillPos, std::vector<T>& lower_bounds, std::vector<T>& priority,T best, lb_strength lb_type)
{
    int _limit1 = node.limit1;
    int _limit2 = node.limit2;

    if(fillPos == node.limit1+1){
        _limit1++;
    }
    else if(fillPos == node.limit2-1){
        _limit2--;
    }else{
        std::cout<<"evaluator : can only bound for children obtained by appending to front or back of partial sequence!\n";
    }

    T costs[2];

    for (int i = node.limit1 + 1; i < node.limit2; i++) {
        int job = node.schedule[i];
        if(mask[job]){
            std::swap(node.schedule[fillPos], node.schedule[i]);
            get_lb(lb_type)->bornes_calculer(node.schedule.data(), _limit1, _limit2, costs, best);
            lower_bounds[job] = costs[0];
            priority[job]=costs[1];
            std::swap(node.schedule[fillPos], node.schedule[i]);
        }
    }
};

//use std::optional...?
/**
 * Compute children bounds using all-child (incremental) bounding function - front, back or both
 *
 *
 */
template<typename T>
void
evaluator<T>::get_children_bounds_incr(subproblem& node, std::vector<T>& lower_bound_begin, std::vector<T>& lower_bound_end, std::vector<T>& priority_begin, std::vector<T>& priority_end, const int begin_end)
{
    switch(begin_end){
        case 0:
        {
            lb->boundChildren(
                node.schedule.data(),node.limit1,node.limit2,
                lower_bound_begin.data(),nullptr,
                priority_begin.data(),nullptr);
            break;
        }
        case 1:
        {
            lb->boundChildren(
                node.schedule.data(),node.limit1,node.limit2,
                nullptr,lower_bound_end.data(),
                nullptr,priority_end.data()
            );
            break;
        }
        case 2:
        {
            lb->boundChildren(
                node.schedule.data(),node.limit1,node.limit2,
                lower_bound_begin.data(),lower_bound_end.data(),
                priority_begin.data(),priority_end.data()
            );
            break;
        }
    }
};






template <typename T>
T
evaluator<T>::get_solution_cost(subproblem& s)
{
    return lb->evalSolution(s.schedule.data());
}


template class evaluator<int>;
