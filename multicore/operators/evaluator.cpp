#include <limits>

#include "evaluator.h"


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
