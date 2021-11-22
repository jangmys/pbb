#include <limits>
#include <algorithm>

#include "branching.h"
#include "evaluator.h"


template<typename T>
void
evaluator<T>::get_children_bounds_strong(subproblem& node, std::vector<int> mask, int begin_end, int fillPos, std::vector<T>& lower_bounds, std::vector<T>& priority,int best)
{
    int _limit1 = node.limit1;
    int _limit2 = node.limit2;
    if(begin_end == branching::Front){
        _limit1++;
    }
    if(begin_end == branching::Back){
        _limit2--; 
    }

    T costs[2];

    for (int i = node.limit1 + 1; i < node.limit2; i++) {
        int job = node.schedule[i];
        if(mask[job]){
            std::swap(node.schedule[fillPos], node.schedule[i]);
            lb2->bornes_calculer(node.schedule.data(), _limit1, _limit2, costs, best);
            lower_bounds[job] = costs[0];
            priority[job]=costs[1];
            std::swap(node.schedule[fillPos], node.schedule[i]);
        }
    }
};

template<typename T>
void
evaluator<T>::get_children_bounds_weak(subproblem& node, std::vector<T>& lower_bound_begin, std::vector<T>& lower_bound_end, std::vector<T>& priority_begin, std::vector<T>& priority_end)
{
    lb->boundChildren(node.schedule.data(),node.limit1,node.limit2,lower_bound_begin.data(),lower_bound_end.data(),priority_begin.data(),priority_end.data());
};




template <typename T>
void
evaluator<T>::getChildrenBounds(subproblem& s, unsigned int fix_at_position, std::vector<T>& LBs, T best_cost)
{
    int costs[2];

    int BE = (fix_at_position == s.limit1+1)?1:0;

    //for all free elements
    for(unsigned int i = s.limit1+1; i<s.limit2; i++)
    {
        //get elem
        int elem = s.schedule[i];

        s.swap(fix_at_position,i);

        lb->bornes_calculer(s.schedule.data(), s.limit1+BE, s.limit2-1+BE, costs, best_cost);

        s.swap(fix_at_position,i);

        LBs[elem] = costs[0];
    }
}

template <typename T>
T
evaluator<T>::getSolutionCost(subproblem& s)
{
    return lb->evalSolution(s.schedule.data());
}


template class evaluator<int>;
