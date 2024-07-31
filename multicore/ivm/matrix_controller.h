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
#include "intervalbb.h"
#include "thread_controller.h"

class IVMController : public ThreadController{
    friend class worker_mc;
public:
    IVMController(pbab* _pbb,int _nthreads,bool distributed = false,int _local_mpi_rank=0);

    int work_share(unsigned id, unsigned thief);

    void initAsEmpty();
    void initFromFac();
    void initFromFac(const unsigned int nbint, const std::vector<int> ids, std::vector<int> pos, std::vector<int> end);
    int getSubproblem(int *ret, const int N);

    bool next();
    void explore_multicore(unsigned id);

    //----------------for distributed mode----------------

    bool is_distributed(){
        return _distributed;
    }

    std::shared_ptr<Intervalbb<int>>get_ivmbb(int k)
    {
        return ivmbb[k];
    };

    pthread_mutex_t mutex_buffer;
private:
    int updatedIntervals = 1;

    std::vector<int> ids;
    std::vector<int> state;
    std::vector<std::vector<int>> pos;
    std::vector<std::vector<int>> end;

    std::vector<std::shared_ptr<Intervalbb<int>>>ivmbb;

    bool _distributed = false;
};

#endif
