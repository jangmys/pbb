#ifndef POOL_CONTROLLER_H_
#define POOL_CONTROLLER_H_

#include <string.h>
#include <thread_controller.h>

#include "pool_thread.h"
#include "poolbb.h"

class bbthread;

class PoolController : public thread_controller{
public:
    PoolController(pbab* pbb,int _nthreads);

    void explore_multicore();

    std::shared_ptr<bbthread> make_bbexplorer();

    int work_share(unsigned id, unsigned dest){
        return 0;
    };

    bool next();

};


#endif
