#include <iostream>
#include <limits>
#include <time.h>

#include <prmu_decompose.h>
// #include <binary_decompose.h>
// #include <tree.h>
#include <shared_pool.h>

#include "arguments.h"
#include "libbounds.h"

#define JOHNSON_BOUND
#define SIMPLE_BOUND

bool all_true(const std::vector<bool>& v)
{
    for(const auto& c : v){
        if(!c)return false;
    }
    return true;
}

template<typename T>
T atomic_read(const T val)
{
    T value;

    #pragma omp atomic read
    value=val;
    return value;
}



int
main(int argc, char ** argv)
{
    //parse cmdline args and .ini file
    //==================================
    arguments::parse_arguments(argc, argv);
    std::cout<<" === solving "<<arguments::problem<<" - instance "<<arguments::inst_name<<std::endl;

    //time everything except argument parsing
    //==================================
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);

    //declare and initialize shared data
    //==================================
    //get num_threads as specified by OMP_NUM_THREADS environment variable
    int nthreads(omp_get_max_threads());
    //make shared pool of permutation subproblems for nthreads threads
    SharedPool<PermutationSubproblem> p(nthreads);
    //flags for termination detection
    std::vector<bool>stop_flags(nthreads,0);
    int end_flag=0;
    //node counters
    int count_decomposed = 0;
    int count_leaves = 0;
    //instance data
    // instance_filename inst(arguments::inst_name);
    instance_taillard inst(arguments::inst_name);

    //global best (solution and cost)
    int global_best_ub=std::numeric_limits<int>::max();
    PermutationSubproblem global_best_solution(inst.size); //not used for now...
    if(arguments::init_mode==0){
        global_best_ub = inst.read_initial_ub_from_file(arguments::inst_name);
        std::cout<<"read UB "<<global_best_ub<<" from file\n";
    }



    //start parallel exploration
    //==========================
    #pragma omp parallel num_threads(nthreads) shared(p,stop_flags,end_flag,inst,global_best_ub) reduction(+:count_decomposed,count_leaves)
    {
        int tid = omp_get_thread_num();
        int nthd = omp_get_num_threads();

        int pbsize;

        //INITIALIZATIONS (thread private data)
        //=====================================
        int local_best = global_best_ub;

        // SIMPLE BOUND
#ifdef SIMPLE_BOUND
        bound_fsp_weak bound;
        #pragma omp critical
        {
            bound.init(&inst);
            pbsize=inst.size;
        }
#endif
        //JOHNSON BOUND
#ifdef JOHNSON
        bound_fsp_strong bound;
        #pragma omp critical
        {
            bound.init(&inst);
            bound.earlyExit=0; //arguments::earlyStopJohnson; ==> don't
            bound.machinePairs=0; //arguments::johnsonPairs; ==> all machine-pairs

            pbsize=inst.size;
        }
#endif

        //all the B&B logic - except exploration - is here (branching,bounding and pruning)
        DecomposePerm decompose(bound);

        //master gets root node
        #pragma omp master
        {
            std::unique_ptr<PermutationSubproblem> root = std::make_unique<PermutationSubproblem>(pbsize);
            p.insert(std::move(root),tid);
        }

        int max_iter=0;

        //should not start before master has initial node (although it might not be a problem)
        #pragma omp barrier

        while(!atomic_read(end_flag)){
            // #pragma omp atomic read
            local_best = atomic_read(global_best_ub);

            std::unique_ptr<PermutationSubproblem> n(p.take(tid));

            if(!n){ //if locally empty and steal failed
                stop_flags[tid]=1;

                if(all_true(stop_flags)){
                    // std::cout<<"thread "<<tid<<" says goodbye "<<p.size(tid)<<"\n";
                    #pragma omp atomic update
                    end_flag++;
                    break;
                }
                continue;
            }else{
                stop_flags[tid]=0;

                count_decomposed++;

                if(n->is_leaf()){
                    int ub = bound.evalSolution(n->schedule.data());
                    if(ub < local_best && ub < atomic_read(global_best_ub)){
                        #pragma omp critical
                        global_best_ub = ub;
                    }
                    ++count_leaves;
                }else{
                    std::vector<std::unique_ptr<PermutationSubproblem>> ns = decompose(*n,local_best);
                    p.insert(std::move(ns),tid);
                }
            }
        }
        #pragma omp critical
        std::cout<<tid<<" decomposed:\t "<<count_decomposed<<"\n";
    }

    //OUTPUT
    //======
    std::cout<<"------------------------------\n";
    std::cout<<count_decomposed<<" nodes decomposed.\n";
    std::cout<<count_leaves<<" leaves.\n";

    std::cout<<"Best Cmax:\t"<<global_best_ub<<"\n";

    clock_gettime(CLOCK_MONOTONIC, &t2);
    std::cout << "time\t" << (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9 << "\n";
} // main
