#ifndef POOL_CONTROLLER_H_
#define POOL_CONTROLLER_H_

#include <assert.h>
#include <string.h>
#include <thread_controller.h>

#include "poolbb.h"


class PoolController : public ThreadController{
public:
    PoolController(pbab* pbb,int _nthreads);

    void explore_multicore();


    int work_share(unsigned id, unsigned thief);

    void set_root(const int id, subproblem& node);

    bool next();

private:
    std::vector<std::shared_ptr<Poolbb>>llbb;
};


#endif
