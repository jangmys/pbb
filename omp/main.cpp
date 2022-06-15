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
// #define SIMPLE_BOUND


template<typename T>
T atomic_read(const T val)
{
    T value;

    #pragma omp flush
    #pragma omp atomic read
    value=val;
    return value;
}

bool all_true(int* flags, unsigned len)
{
    for(unsigned i=0;i<len;++i){
        // if(!flags[i])return false;
        if(!atomic_read(flags[i]))return false;
    }
    return true;
}


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
    SharedPool<PermutationSubproblem> p;
    //flags for termination detection
    // std::vector<bool>stop_flags(nthreads,0); //some problem with atomic operations...compiler doesn't like it
    int *stop_flags = new int[nthreads];

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

    //master gets root node
    std::unique_ptr<PermutationSubproblem> root = std::make_unique<PermutationSubproblem>(inst.size);
    p.insert(std::move(root),0);

    //start parallel exploration
    //==========================
    #pragma omp parallel num_threads(nthreads) reduction(+:count_decomposed,count_leaves)
    {
        int tid = omp_get_thread_num();

        //INITIALIZATIONS (thread private data)
        //=====================================
        int local_best = global_best_ub;

        // SIMPLE BOUND
#ifdef SIMPLE_BOUND
        bound_fsp_weak bound;
        #pragma omp critical
        {
            bound.init(&inst);
        }
#endif
        //JOHNSON BOUND
#ifdef JOHNSON_BOUND
        bound_fsp_strong bound;
        #pragma omp critical
        {
            bound.init(&inst);
            bound.earlyExit=0; //arguments::earlyStopJohnson; ==> don't
            bound.machinePairs=0; //arguments::johnsonPairs; ==> all machine-pairs
        }
#endif

        #pragma omp atomic write
        stop_flags[tid]=0;
        #pragma omp flush

        //all the B&B logic - except exploration - is here (branching,bounding and pruning)
        DecomposePerm decompose(bound);

        while(true){
            #pragma omp atomic read
            local_best = global_best_ub;
            std::unique_ptr<PermutationSubproblem> n(p.take(tid));

            if(!n){ //if locally empty and steal failed
                #pragma omp atomic write
                stop_flags[tid]=1;
                #pragma omp flush

                if(all_true(stop_flags,omp_get_num_threads())){
                    break;
                }
                continue;
            }else{
                #pragma omp atomic write
                stop_flags[tid]=0;
                #pragma omp flush

                count_decomposed++;

                if(n->is_leaf()){
                    int ub = bound.evalSolution(n->schedule.data());
                    if(ub < local_best && ub < atomic_read(global_best_ub)){
                        #pragma omp critical
                        {
                            //should update permutation as well...
                            global_best_ub = ub;
                        }
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

    delete[] stop_flags;

    //OUTPUT
    //======
    std::cout<<"------------------------------\n";
    std::cout<<count_decomposed<<" nodes decomposed.\n";
    std::cout<<count_leaves<<" leaves.\n";

    std::cout<<"Best Cmax:\t"<<global_best_ub<<"\n";

    clock_gettime(CLOCK_MONOTONIC, &t2);
    std::cout << "time\t" << (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9 << "\n";
} // main
