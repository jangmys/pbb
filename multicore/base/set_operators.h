#ifndef SET_OPERATORS_H_
#define SET_OPERATORS_H_

#include <iostream>

#include "pruning.h"
#include "branching.h"
// #include <mcbb.h>

//choose pruning operator of a MCbb (IVM or Pool) from pbab and arguments
template<typename T>
std::unique_ptr<Pruning> make_prune_ptr(T initial_cost) //pbab* pbb)
{
    if(!arguments::findAll){
        //keepSmaller <==> prune if LB equal to best
        return std::make_unique<keepSmaller>(initial_cost);
    }else{
        //keepEqualOrSmaller <==> prune if LB worse than best
        return std::make_unique<keepEqualOrSmaller>(initial_cost);
    }
    return nullptr;
};

//choosebranching operator of a MCbb (IVM or Pool) from pbab and arguments
template<typename T>
std::unique_ptr<Branching> make_branch_ptr(int size, T initial_cost)
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
            return std::make_unique<maxSumBranching>(size);
        }
        case 2:{
            return std::make_unique<minBranchBranching>(size,initial_cost);
        }
        case 3:{
            return std::make_unique<minMinBranching>(size,initial_cost);
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
                bd->init(*(pbb->inst.get()));
                return bd;
            }
            case 1:
            {
                auto bd = std::make_unique<bound_fsp_strong>();
                bd->init(*(pbb->inst.get()));
                bd->earlyExit=arguments::earlyStopJohnson;
                bd->machinePairs=arguments::johnsonPairs;
                return bd;
            }
        }
    }else if(arguments::problem[0]=='d'){
        std::cout<<"Dummy bound\n"<<std::endl;
        auto bd = std::make_unique<bound_dummy>();
        bd->init(*(pbb->inst.get()));
        return bd;
    }else{
        std::cout<<"no valid problem defined. can't continue.\n";
        return nullptr;
    }
    return nullptr;
}




#endif
