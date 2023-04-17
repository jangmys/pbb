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

class gpubb {
public:
    gpubb(pbab * pbb);
    ~gpubb();

    void
    initializeGPU(int _cost);

    std::atomic<bool> allEnd;
    pthread_mutex_t mutex_end;

    void interruptExploration();
    void initFullInterval();

    pbab * pbb;
    int nbIVM;
    int size;
    int initialUB;

    bool localFoundNew;

    executionmode execmode;

    // bound_abstract<int> * bound;
    std::unique_ptr<bound_abstract<int>> bound ;

    int topoDimensions;
    int topoA[MAX_HYPERCUBE_DIMS];
    int topoB[MAX_HYPERCUBE_DIMS];
    int topoRings[MAX_HYPERCUBE_DIMS];
    void
    setHypercubeConfig();

    cudaStream_t stream;
    cudaEvent_t event;

    int id;

    struct timespec starttime;

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

    int * todo_d;
    int * tmp_arr_d, * auxArr;
    int * auxEnd;

    int * ivmId_d;
    int * toSwap_d;

    unsigned int* depth_histo_h;
    unsigned int* depth_histo_d;

    // global counters
    unsigned int * counter_h, * counter_d;
    unsigned int * ctrl_d, * ctrl_h;

    unsigned long long int * nbDecomposed_h, * nbDecomposed_d;
    unsigned int * nbLeaves_d;

    int * flagLeaf;

    // WORK STEALING
    int * split_d; // place to cut interval
    int * victim_flag;
    int * victim_h, * victim_d;         // map thief->victim
    int * length_h, * length_d;         // length of intervals
    int * sumLength_h, * sumLength_d;   // sum of interval-lengths
    int * meanLength_h, * meanLength_d; // average interval-length

    int ringsize; // for multi-D topology

    int ws_granular; // granularity
    int search_from;
    int search_to;
    int search_step; // parameters for extended ring-strategy
    float search_cut;

    bool startclock;
    bool firstbound;

    //for _FSP_
    int * front_h, * front_d;
    int * back_h, * back_d;
    void initializeBoundFSP();

    //for TEST
    void copyH2DconstantTEST();
    void initializeBoundTEST();

    void getExplorationStats(const int,const int);
    void selectAndBranch(const int);
    void launchBBkernel(const int);

    bool allDone();

    bool decode(const int NN);
    bool weakBound(const int NN, const int best);

    void buildMapping();

    void allocate_on_host();
    void allocate_on_device();

    void free_on_host();
    void free_on_device();

    void copyH2D();
    void copyH2D_update();
    void copyH2Dconstant();
    void copyD2H();


    bool boundLeaves(bool reached, int &best);

    // ===========================
    void initialize();


    void initializeIVM(bool root, int id);
    void initAtInterval(int *, int *);
    int getIntervals(std::shared_ptr<fact_work> fwrk);

    int getDeepSubproblem(int *ret, const int N);

    bool next();
    bool next(int &, int &);
    bool triggeredNext(int& best, int& iter);

    int steal_in_device(int);
    void adapt_workstealing(int, int);

    void affiche(int M);
    void getStats();

    //for worker-mode (distributed) only
    void initFromFac(const int nbint, const int* ids, int*pos, int* end);
    void getIntervals(int *pos, int* end, int *ids, unsigned &nb_intervals, const int max_intervals);
};
#endif
