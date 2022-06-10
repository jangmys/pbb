#ifndef PRMU_DECOMPOSE_H
#define PRMU_DECOMPOSE_H

#include <vector>
#include <chrono>
#include <thread>

#include <base_decompose.h>
#include <prmu_subproblem.h>

class DecomposePerm : public DecomposeBase<PermutationSubproblem>
{
public:
    DecomposePerm(bound_abstract<int>& _eval):
        DecomposeBase(_eval)
    {};

    std::vector<std::unique_ptr<PermutationSubproblem>> operator()(PermutationSubproblem& n, const int best_ub){
        std::vector<std::unique_ptr<PermutationSubproblem>>children;

        // std::cout<<"Parent:\t"<<n<<std::endl;

       //reverse (to get lexicographic DFS)
       for (int j = n.limit2 - 1; j > n.limit1; j--) {
           //generates j^th child of parent node n
           //...maybe just copy construct and modify child here (feels weird to have branching logic hidden in ctor...)
           auto tmp = std::make_unique<PermutationSubproblem>(n,j);

           int costs[2];
           eval.bornes_calculer(tmp->schedule.data(),tmp->limit1,tmp->limit2,costs,best_ub);

           if(costs[0]<best_ub){
               tmp->lb_value = costs[0];
               children.push_back(std::move(tmp));
           }

           // #pragma omp critical
           // std::cout<<*tmp<<"\t"<<costs[0]<<std::endl;
           // std::this_thread::sleep_for(std::chrono::nanoseconds(100));
       }

       return children;
    };
};

#endif
