/*
named constructor for IVM-based BB
*/
#ifndef MAKE_POOL_BB_H
#define MAKE_POOL_BB_H

#include <memory>
#include <arguments.h>
#include <pbab.h>

#include "../base/mcbb.h"
#include "poolbb.h"

std::shared_ptr<Poolbb> make_poolbb(pbab* pbb);

#endif
