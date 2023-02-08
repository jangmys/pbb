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
    DecomposePerm(std::unique_ptr<bound_abstract<int>> _eval):
        DecomposeBase(std::move(_eval))
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
           eval->bornes_calculer(tmp->schedule.data(),tmp->limit1,tmp->limit2,costs,best_ub);

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



class DecomposePermIncr : public DecomposeBase<PermutationSubproblem>
{
public:
    DecomposePermIncr(std::unique_ptr<bound_abstract<int>> _eval):
        DecomposeBase(std::move(_eval))
    {};

    std::vector<std::unique_ptr<PermutationSubproblem>> operator()(PermutationSubproblem& n, const int best_ub){
        std::vector<std::unique_ptr<PermutationSubproblem>>children;

        std::vector<int> costFwd(n.size);
        std::vector<int> costBwd(n.size);

        std::vector<int> prioFwd(n.size);
        std::vector<int> prioBwd(n.size);

        //evaluate lower bounds and priority
        eval->boundChildren(n.schedule.data(),n.limit1,n.limit2, costFwd.data(),costBwd.data(), prioFwd.data(),prioBwd.data(), 99999);

        for (int j = n.limit2 - 1; j > n.limit1; j--) {
            int job = n.schedule[j];

            if(costFwd[job]<best_ub){
                auto tmp = std::make_unique<PermutationSubproblem>(n,j);
                tmp->lb_value=costFwd[job];
                children.push_back(std::move(tmp));
            }
        }

       return children;
    };
};



#endif
