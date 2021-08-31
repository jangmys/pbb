#ifndef MATRIX_CONTROLLER_H
#define MATRIX_CONTROLLER_H

#include <sys/sysinfo.h>
#include <pthread.h>
#include <atomic>
#include <memory>
#include <vector>
#include <list>
#include <deque>

#include "ivmthread.h"
#include "thread_controller.h"

class pbab;
class bbthread;

// static pthread_mutex_t instance_mutex = PTHREAD_MUTEX_INITIALIZER;

class matrix_controller : public thread_controller{
public:
    ivmthread* make_bbexplorer(unsigned _id){
        //initialize local (sequential) BB
        pthread_mutex_lock(&instance_mutex);
        ivmthread* ibb = new ivmthread(pbb);
        pthread_mutex_unlock(&instance_mutex);
        return ibb;
    }
    int work_share(unsigned id, unsigned thief);

    matrix_controller(pbab* _pbb) : thread_controller(_pbb){
        for (int i = 0; i < (int) M; i++){
            // victim_list.push_back(i);
            bbb[i]=NULL;
            // sbb[i]=NULL;
        }

        resetExplorationState();

        state = std::vector<int>(M,0);
        root = std::vector<int>(size,0);
        for(int i=0;i<size;i++){
            root[i]=pbb->root_sltn->perm[i];
        }
        for(unsigned i=0;i<M;i++){
            pos.emplace_back(std::vector<int>(size,0));
            end.emplace_back(std::vector<int>(size,0));
        }

        std::cout<<"+++ done \n"<<std::endl;
    };

    ~matrix_controller()
    {
        pthread_mutex_destroy(&mutex_steal_list);
        // pthread_mutex_destroy(&mutex_end);
    };

	std::vector<int> ids;
	std::vector<int> state;
	std::vector<std::vector<int>> pos;
	std::vector<std::vector<int>> end;

	std::vector<int>root;

    int updatedIntervals = 1;

	static bool first;

    // std::atomic<int> foundNew{0};

    void initFullInterval();
    void initFromFac(const unsigned int nbint, const int* ids, int*pos, int* end);

    void getIntervals(int *pos, int* end, int *ids, int &nb_intervals, const unsigned int max_intervals);
    int getNbIVM();
    int getSubproblem(int *ret, const int N);

    bool solvedAtRoot();

    void allocate();

    void unlockWaiting(unsigned id);
    void resetExplorationState();
    void interruptExploration();


    bool next();
    void printStats();

    void explore_multicore();
};

#endif
