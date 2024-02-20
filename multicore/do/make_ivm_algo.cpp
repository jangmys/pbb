#include <arguments.h>
#include <pbab.h>

#include "make_ivm_algo.h"

//factory for interval-based CPU-BB : build and configure CPU-BB according to arguments
template<typename T>
std::shared_ptr<Intervalbb<T>> make_ivmbb(pbab* pbb)
{
    std::shared_ptr<Intervalbb<T>> bb;

    if(arguments::boundMode == 0){
        bb = std::make_shared<Intervalbb<T>>(pbb);
        bb->set_primary_bound( make_bound_ptr<int>(pbb,arguments::primary_bound));
    }else if(arguments::boundMode == 1){
        bb = std::make_shared<IntervalbbEasy<T>>(pbb);
        bb->set_primary_bound( make_bound_ptr<int>(pbb,arguments::primary_bound));
    }else{
        bb = std::make_shared<IntervalbbIncr<T>>(pbb);
        bb->set_primary_bound( make_bound_ptr<int>(pbb,arguments::primary_bound));
        bb->set_secondary_bound( make_bound_ptr<int>(pbb,arguments::secondary_bound));
    }

    bb->set_prune( make_prune_ptr<int>(pbb->best_found.initial_cost) );
    bb->set_branch( make_branch_ptr<int>(pbb->size,pbb->best_found.initial_cost) );

    bb->print_new_solutions = arguments::printSolutions;

    return bb;
}

template std::shared_ptr<Intervalbb<int>> make_ivmbb<int>(pbab* pbb);
