#include <iostream>
#include <limits>
#include <time.h>

#include <prmu_decompose.h>
// #include <binary_decompose.h>
// #include <tree.h>
#include <shared_pool.h>

#include "arguments.h"
#include "libbounds.h"

#include "make_bound.h"


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
    if(argc<5){
        std::cout<<"3 arguments expected : \n";
        std::cout<<"1) -z p=fsp,i=ta10,o\n";
        std::cout<<"2) ['s','j'] : simple/johnson\n";
        std::cout<<"3) [0,1] : one-by-one / incremental\n";
        exit(1);
    }

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
    long long int count_decomposed = 0;
    long long int count_leaves = 0;

    //instance data
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

    //bound
    arguments::earlyStopJohnson = false;
    arguments::johnsonPairs = 0;

    //start parallel exploration
    //==========================
    #pragma omp parallel num_threads(nthreads) reduction(+:count_decomposed,count_leaves)
    {
        //INITIALIZATIONS (thread private data)
        //=====================================
        int tid = omp_get_thread_num();

        int local_stop = 0;
        int local_best = global_best_ub;

        #pragma omp atomic write
        stop_flags[tid]=0;
        #pragma omp flush

        // evaluation function (private for performance reasons)
        std::unique_ptr<bound_abstract<int>> bound;
        switch(argv[3][0]){
            case 's': // SIMPLE BOUND
            {
                bound = make_bound_ptr<int>(inst,0);
                break;
            }
            case 'j': //JOHNSON
            {
                bound = make_bound_ptr<int>(inst,1);
                break;
            }
        }

        //all the B&B logic - except exploration - is here (branching,bounding and pruning)
        std::unique_ptr<DecomposeBase<PermutationSubproblem>> decompose;
        switch(atoi(argv[4])){
            case 0:
            {
                decompose = std::make_unique<DecomposePerm>(std::move(bound));
                break;
            }
            case 1:
            {
                decompose = std::make_unique<DecomposePermIncr>(std::move(bound));
                break;
            }
        }

        // Main exploration loop
        //=========================================
        while(true){
            #pragma omp atomic read
            local_best = global_best_ub;
            std::unique_ptr<PermutationSubproblem> n(p.take(tid));

            if(!n){ //if locally empty and steal failed
                local_stop = 1;

                #pragma omp atomic write
                stop_flags[tid]=local_stop;
                #pragma omp flush

                if(all_true(stop_flags,omp_get_num_threads())){
                    break;
                }
                continue;
            }else{
                if(local_stop){
                    local_stop = 0;
                    #pragma omp atomic write
                    stop_flags[tid]=0;
                    #pragma omp flush
                }
                count_decomposed++;

                if(n->is_leaf()){
                    int ub = bound->evalSolution(n->schedule.data());
                    if(ub < local_best && ub < atomic_read(global_best_ub)){
                        #pragma omp critical
                        {
                            //should update permutation as well...
                            global_best_ub = ub;
                        }
                    }
                    ++count_leaves;
                }else{
                    std::vector<std::unique_ptr<PermutationSubproblem>> ns = (*decompose)(*n,local_best);
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
