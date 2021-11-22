#ifndef MATRIX_CONTROLLER_H
#define MATRIX_CONTROLLER_H

#include <sys/sysinfo.h>
#include <pthread.h>
#include <atomic>
#include <memory>
#include <vector>
#include <list>
#include <deque>

#include "macros.h"
#include "ivmthread.h"
#include "thread_controller.h"

class pbab;
class bbthread;

class matrix_controller : public thread_controller{
public:
    ivmthread* make_bbexplorer(unsigned _id);
    int work_share(unsigned id, unsigned thief);

    matrix_controller(pbab* _pbb);
    ~matrix_controller();

	std::vector<int> ids;
	std::vector<int> state;
	std::vector<std::vector<int>> pos;
	std::vector<std::vector<int>> end;

	std::vector<int>root;

    int updatedIntervals = 1;

    void initFullInterval();
    void initFromFac(const unsigned int nbint, const int* ids, int*pos, int* end);

    void getIntervals(int *pos, int* end, int *ids, int &nb_intervals, const unsigned int max_intervals);
    int getNbIVM();
    int getSubproblem(int *ret, const int N);

    bool solvedAtRoot();


    void unlockWaiting(unsigned id);
    void resetExplorationState();
    void interruptExploration();

    bool next();
    void printStats();

    void explore_multicore();
};

#endif
