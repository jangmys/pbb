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

    int nProc;

    works* wrks;
    communicator* comm;

    std::shared_ptr<work> wrk;//(new work(pbb));

    bool globalEnd;
    bool first;
    bool isSharing;

    master(pbab* _pbb);
    ~master();

    void reset();

    void initWorks(int initMode);
    int processRequest(std::shared_ptr<work> w);
    void shutdown();
    void run();
};

#endif
