#include "pool_thread.h"

PoolThread::PoolThread(pbab* pbb) : bbthread(pbb){};

int
PoolThread::shareWork(std::shared_ptr<PoolThread> thief_thread)
{

}
