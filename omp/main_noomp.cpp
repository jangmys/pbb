#include <iostream>
#include <time.h>

#include <prmu_decompose.h>
#include <single_pool.h>


#include "arguments.h"
#include "libbounds.h"


bool all_true(const std::vector<bool>& v)
{
    for(const auto& c : v){
        if(!c)return false;
    }
    return true;
}


int
main(int argc, char ** argv)
{
    //******************************
    arguments::parse_arguments(argc, argv);
    std::cout<<" === solving "<<arguments::problem<<" - instance "<<arguments::inst_name<<std::endl;
    //******************************

    // instance_filename inst(arguments::inst_name);
    instance_taillard inst(arguments::inst_name);

    bool early_stop = false;
    int machine_pairs = 0;
    BoundFactory bound_factory(
        std::make_unique<instance_taillard>(inst),
        early_stop,
        machine_pairs
    );

    auto bound = bound_factory.make_bound(0);

    // bound_fsp_weak bound;
    bound->init(&inst);

    DecomposePerm decompose(std::move(bound));



    int SIZE = 10;

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);

    SinglePool<PermutationSubproblem> p;

    int count_decomposed = 0;
    int count_leaves = 0;

    int best_ub = 9999999;


    std::unique_ptr<PermutationSubproblem> root = std::make_unique<PermutationSubproblem>(SIZE);
    p.insert(std::move(root));

    while(1){
        std::unique_ptr<PermutationSubproblem> n(p.take());

        if(!n){ //if locally empty and steal failed
            break;
        }else{
            count_decomposed++;
            if(n->is_leaf()){
                ++count_leaves;
            }else{
                std::vector<std::unique_ptr<PermutationSubproblem>> ns = decompose(*n,best_ub);
                p.insert(std::move(ns));
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t2);
    std::cout << "time\t" << (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9 << "\n";
} // main
