#ifndef MASTER_H
#define MASTER_H

#include <atomic>
#include <memory>
#include <climits>
#include <string.h>

#include "gmp.h"
#include "gmpxx.h"

class pbab;
class work;
class works;
class communicator;

class master{
public:
    pbab* pbb;

    unsigned int work_in;
    unsigned int work_out;
    int nProc;

    works* wrks;
    // solution* master_sol;
    communicator* comm;

    std::shared_ptr<work> wrk;//(new work(pbb));

    bool end;
    bool first;
    bool stopSharing;
    // bool foundSolution;


    master(pbab* _pbb);
    ~master();

    void reset();

    void initWorks(int initMode);
    bool processRequest(std::shared_ptr<work> w);//, bool &shutdownWorker);
    void shutdown();
    void run();
    // void test();
};

#endif
