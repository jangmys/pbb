#ifndef GPUBB_H
#define GPUBB_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <memory>
#include <atomic>

#include "libbounds.h"
#include "./gpuerrchk.h"

#define MAX_HYPERCUBE_DIMS 20

class pbab;
class fact_work;

//to distinguish between alternative implementations... (to complete)
struct executionmode
{
    bool triggered;
};


class gpu_worksteal
{
public:
    static constexpr size_t MaxHypercubeDims=20;

    size_t N;
    size_t nbIVM;

    float search_cut;
    int topoDimensions;
    int topoA[MaxHypercubeDims];
    int topoB[MaxHypercubeDims];
    int topoRings[MaxHypercubeDims];

    int *length_d; // length of intervals (factoradic)
    int *meanLength_d;
    int *sumLength_d;   // sum of interval-lengths
    int *victim_d;
    int *victim_flag_d;

    gpu_worksteal(int _N, int _nbIVM) : N(_N),nbIVM(_nbIVM),search_cut(1.0){
        std::cout<<"init ws with "<<nbIVM<<std::endl;
        gpuErrchk(cudaMalloc((void **) &length_d, N*nbIVM*sizeof(int)));
        gpuErrchk(cudaMalloc((void **) &victim_d, nbIVM*sizeof(int)));
        gpuErrchk(cudaMalloc((void **) &victim_flag_d, nbIVM*sizeof(int)));
        gpuErrchk(cudaMalloc((void **) &meanLength_d, N*sizeof(int)));
        gpuErrchk(cudaMalloc((void **) &sumLength_d, N*sizeof(int)));

        switch (nbIVM) {
            case 1:
                topoDimensions = 1;
                topoA[0]       = 0;
                topoB[0]       = 0;
                break;

            case 4:
                topoDimensions = 1;
                topoA[0]       = 0;
                topoB[0]       = 2;
                break;

            case 64: // 64 == 4*4*4
                topoDimensions = 3;
                topoA[0]       = 0; topoA[1]       = 2; topoA[2]       = 4;
                topoB[0]       = 2; topoB[1]       = 2; topoB[2]       = 2;
                break;

            case 128: // 64 == 4*4*4
                topoDimensions = 3;
                topoA[0]       = 0;            topoA[1]       = 2;            topoA[2]       = 4;
                topoB[0]       = 2;            topoB[1]       = 2;            topoB[2]       = 3;
                break;

            case 512: // 512 == 4*4*4*8
            {
                topoDimensions = 4;
                for(int i=0;i<topoDimensions;i++){
                    topoA[i]=2*i;
                    topoB[i]=2;
                }
                topoB[3] = 3;
                break;
            }
            case 1024: // 2**(2 2 2 2 2)
            {
                topoDimensions = 5;
                for(int i=0;i<topoDimensions;i++){
                    topoA[i]=2*i;
                    topoB[i]=2;
                }
                break;
            }
            case 2048: // 8*8*8*4 == 2**(3+3+3+2)
            {
                topoDimensions = 5;
                for(int i=0;i<topoDimensions;i++){
                    topoA[i]=2*i;
                    topoB[i]=2;
                }
                topoB[4] = 3;
                break;
            }
            case 4096: // 4096 == 8**4 (2**12)
            {
                // ws.topoDimensions = 6;
                topoDimensions = 6;
                for(int i=0;i<topoDimensions;i++){
                    topoA[i]=2*i;
                    topoB[i]=2;
                }
                break;
            }
            case 8192: // 4096 == 16 * 8**3 (2**13)
            {
                topoDimensions = 6;
                for(int i=0;i<topoDimensions;i++){
                    topoA[i]=2*i;
                    topoB[i]=2;
                }
                topoB[5]       = 3;
                break;
            }
            case 16384: // (2**14)
            {
                topoDimensions = 7;
                for(int i=0;i<topoDimensions;i++){
                    topoA[i]=2*i;
                    topoB[i]=2;
                }
                break;
            }
            case 32768: // 32768 == 8**5
            {
                topoDimensions = 7;
                for(int i=0;i<topoDimensions;i++){
                    topoA[i]=2*i;
                    topoB[i]=2;
                }
                topoB[6]       = 3;
                break;
            }
            case 65536: // 65536 == 2 * 8**5 (2**16)
            {
                topoDimensions = 8;
                for(int i=0;i<topoDimensions;i++){
                    topoA[i]=2*i;
                    topoB[i]=2;
                }
                break;
            }
            case 131072: // 2**17
            {
                topoDimensions = 8;
                for(int i=0;i<topoDimensions;i++){
                    topoA[i]=2*i;
                    topoB[i]=2;
                }
                topoB[7]       = 3;
                break;
            }    exit(0);

            default:
                std::cout<<"gpu_worksteal : invalid ivm-nb\n"<<std::endl;
                exit(0);
                break;
        }
    }

    ~gpu_worksteal(){
        gpuErrchk(cudaFree(length_d));
        gpuErrchk(cudaFree(meanLength_d));
        gpuErrchk(cudaFree(sumLength_d));
        gpuErrchk(cudaFree(victim_d));
        gpuErrchk(cudaFree(victim_flag_d));
    }

    void adapt_workstealing(unsigned int nb_exploring, unsigned int min, unsigned int max)
    {
        if ((nb_exploring >= (unsigned int) (7 * nbIVM / 10)) && (search_cut < 0.8))
            search_cut += 0.1;
        else if ((nb_exploring < (unsigned int) (7 * nbIVM / 10)) && (search_cut > 0.2))
            search_cut -= 0.1;

    	FILE_LOG(logDEBUG) << "GPUWS length coefficient: "<<search_cut;
    }

    int
    steal_in_device(int* line, int* pos, int* end, int* dir, int* mat, int* state, int iter, unsigned int nb_exploring);
};

class gpu_fsp_bound{
public:
    size_t N;
    size_t M;
    size_t nbIVM;

    int* front_d;
    int* back_d;

    gpu_fsp_bound(size_t _N, size_t _M, size_t _nbIVM) : N(_N),M(_M),nbIVM(_nbIVM){
        gpuErrchk(cudaMalloc((void **) &front_d, nbIVM * M * sizeof(int)));
        gpuErrchk(cudaMalloc((void **) &back_d, nbIVM * M * sizeof(int)));
    }

    ~gpu_fsp_bound(){
        gpuErrchk(cudaFree(front_d));
        gpuErrchk(cudaFree(back_d));
    }
};

class gpu_ivm{
public:
    size_t N;
    size_t nbIVM;

    int *mat_d, *mat_h;
    int *pos_d, *pos_h;
    int *end_d, *end_h;
    int *dir_d, *dir_h;

    gpu_ivm(size_t _N, size_t _nbIVM) : N(_N),nbIVM(_nbIVM){
        gpuErrchk(cudaMalloc((void **) &pos_d, nbIVM * N * N * sizeof(int)));
        gpuErrchk(cudaMalloc((void **) &pos_d, nbIVM * N * sizeof(int)));
        gpuErrchk(cudaMalloc((void **) &end_d, nbIVM * N * sizeof(int)));
        gpuErrchk(cudaMalloc((void **) &dir_d, nbIVM * N * sizeof(int)));
    }
};



class gpubb {
public:
    pbab * pbb;
    int size;
    int nbIVM;
    gpu_worksteal ws;

    gpubb(pbab * pbb);
    ~gpubb();

    void
    initializeGPU(int _cost);

    int initialUB;
    std::atomic<bool> allEnd;
    pthread_mutex_t mutex_end;

    void interruptExploration();
    void initFullInterval();

    bool localFoundNew;

    executionmode execmode;

    // bound_abstract<int> * bound;
    std::unique_ptr<bound_abstract<int>> bound ;


    // ==============================================

    // ==============================================
    cudaStream_t *stream;
    cudaEvent_t *event;
    // ==============================================



    // integers with values <= SIZE  //device,host
    int * mat_d, * mat_h;
    int * pos_d, * pos_h;
    int * end_d, * end_h;
    int * dir_d, * dir_h;
    int * line_d, * line_h;
    int * state_d, * state_h;
    int * lim1_d, * lim1_h;
    int * lim2_d, * lim2_h;
    int * schedule_h, * schedule_d;

    int * costsBE_h, * costsBE_d;
    int * prio_d;
    int * sums_d;

    // global counters
    unsigned int * counter_h, * counter_d;
    unsigned int * ctrl_d, * ctrl_h;

    unsigned long long int * nbDecomposed_h, * nbDecomposed_d;
    unsigned int * nbLeaves_d;

    int * flagLeaf;

    // =======================================================================
    // WORK STEALING
    int * split_d; // place to cut interval
    int * victim_flag;
    int * victim_h, * victim_d;         // map thief->victim
    int * length_h, * length_d;         // length of intervals
    int * sumLength_h, * sumLength_d;   // sum of interval-lengths
    int * meanLength_h, * meanLength_d; // average interval-length

    // int search_from;
    // int search_to;
    // int search_step; // parameters for extended ring-strategy
    // float search_cut;
    // =======================================================================

    bool firstbound;

    int * todo_d;
    int * tmp_arr_d, * auxArr;
    int * auxEnd;

    int * ivmId_d;
    int * toSwap_d;

    //for flowshop
    std::unique_ptr<gpu_fsp_bound> bd;

    // int * front_h, * front_d;
    // int * back_h, * back_d;
    void initializeBoundFSP();
    bool weakBound(const int NN, const int best);
    void buildMapping(int best);
    bool boundLeaves(bool reached, int &best);


    //for TEST
    void copyH2DconstantTEST();
    void initializeBoundTEST();




    void getExplorationStats(const int,const int);
    void selectAndBranch(const int);
    void launchBBkernel(const int);

    bool allDone();
    bool decode(const int NN);

    void allocate_on_host();
    void allocate_on_device();

    void free_on_host();
    void free_on_device();

    void copyH2D();
    void copyH2D_update();
    void copyH2Dconstant();
    void copyD2H();



    // ===========================
    void initialize(int rank);


    void initializeIVM(bool root, int id);
    void initAtInterval(int *, int *);
    int getIntervals(std::shared_ptr<fact_work> fwrk);

    int getDeepSubproblem(int *ret, const int N);

    bool next();
    bool next(int &, int);
    bool triggeredNext(int& best, int iter);

    int steal_in_device(int* line, int* pos, int* end, int* dir, int* mat, int* state, int iter);
    // void adapt_workstealing(unsigned int, unsigned int, unsigned int);

    void affiche(int M);
    void getStats();

    //for worker-mode (distributed) only
    void initFromFac(const int nbint, const int* ids, int*pos, int* end);
    void getIntervals(int *pos, int* end, int *ids, unsigned &nb_intervals, const int max_intervals);
};
#endif
