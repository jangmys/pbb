#ifndef SET_OPERATORS_H_
#define SET_OPERATORS_H_

#include "operators/pruning.h"
// #include <mcbb.h>

//choose pruning operator of a MCbb (IVM or Pool) from pbab and arguments
template<typename T>
std::unique_ptr<Pruning> make_prune_ptr(pbab* pbb)
{
    if(arguments::findAll){
        return std::make_unique<keepSmaller>(pbb->best_found.initial_cost);
    }else{
        return std::make_unique<keepEqualOrSmaller>(pbb->best_found.initial_cost);
    }
    return nullptr;
};

//choosebranching operator of a MCbb (IVM or Pool) from pbab and arguments
template<typename T>
std::unique_ptr<Branching> make_branch_ptr(pbab* pbb)
{
    switch (arguments::branchingMode) {
        case -3:{
            return std::make_unique<alternateBranching>();
        }
        case -2:{
            return std::make_unique<forwardBranching>();
        }
        case -1:{
            return std::make_unique<backwardBranching>();
        }
        case 1:{
            return std::make_unique<maxSumBranching>(pbb->size);
        }
        case 2:{
            return std::make_unique<minBranchBranching>(pbb->size,pbb->best_found.initial_cost);
        }
        case 3:{
            return std::make_unique<minMinBranching>(pbb->size,pbb->best_found.initial_cost);
        }
        default:{
            printf("branching rule not defined\n");
            return nullptr;
        }
    }
}

template<typename T>
std::unique_ptr<bound_abstract<int>> make_bound_ptr(pbab* pbb, const int _bound_choice = 0)
{
    if(arguments::problem[0]=='f'){
        switch (_bound_choice) {
            case 0:
            {
                auto bd = std::make_unique<bound_fsp_weak>();
                bd->init(pbb->inst);
                return bd;
            }
            case 1:
            {
                auto bd = std::make_unique<bound_fsp_strong>();
                bd->init(pbb->inst);
                bd->earlyExit=arguments::earlyStopJohnson;
                bd->machinePairs=arguments::johnsonPairs;
                return bd;
            }
        }
    }else if(arguments::problem[0]=='d'){
        auto bd = std::make_unique<bound_dummy>();
        bd->init(pbb->inst);
        return bd;
    }
    return nullptr;
}




#endif
