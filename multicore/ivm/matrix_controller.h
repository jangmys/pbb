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
    friend class worker_mc;
public:
    explicit matrix_controller(pbab* _pbb,int _nthreads);

    std::shared_ptr<bbthread> make_bbexplorer();
    int work_share(unsigned id, unsigned thief);

    void initFullInterval();
    void initFromFac(const unsigned int nbint, const int* ids, int*pos, int* end);
    int getSubproblem(int *ret, const int N);

    void unlockWaiting(unsigned id);

    bool next();
    void explore_multicore();

    void set_distributed(){
        _distributed = true;
    }
    bool is_distributed(){
        return _distributed;
    }

private:
    int updatedIntervals = 1;

    std::vector<int> ids;
    std::vector<int> state;
    std::vector<std::vector<int>> pos;
    std::vector<std::vector<int>> end;

    bool _distributed = false;
};

#endif
