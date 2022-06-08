#ifndef INTERVALBB_INCR_H_
#define INTERVALBB_INCR_H_

#include "intervalbb.h"

#include "libbounds.h"

class pbab;

template<typename T>
class IntervalbbIncr : public Intervalbb<T>{
public:
    IntervalbbIncr(pbab* _pbb) : Intervalbb<T>(_pbb){};



};

#endif
