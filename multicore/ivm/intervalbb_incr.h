#ifndef INTERVALBB_INCR_H_
#define INTERVALBB_INCR_H_

#include "intervalbb.h"

#include "libbounds.h"

class pbab;

template<typename T>
class IntervalbbIncr : public Intervalbb<T>{
public:
    IntervalbbIncr(pbab* _pbb,
        std::unique_ptr<Branching> _branch,
        std::unique_ptr<Pruning> _prune
    ) : Intervalbb<T>(_pbb,std::move(_branch),std::move(_prune)){};

    

};

#endif
