#ifndef MASTER_H
#define MASTER_H

#include <atomic>
#include <memory>
#include <climits>
#include <string.h>

#include "gmp.h"
#include "gmpxx.h"

#include "pbab.h"
#include "work.h"
#include "works.h"
#include "communicator.h"

class master{
    pbab* pbb;
    communicator comm;
    works wrks;
    std::shared_ptr<work> wrk;
    bool isSharing;
    int nProc;
public:
    master(pbab* _pbb);

    void reset();

    void initWorks(int initMode);
    int processRequest(std::shared_ptr<work> w);
    void shutdown();
    void run();
};

#endif
