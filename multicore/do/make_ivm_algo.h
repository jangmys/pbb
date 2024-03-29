/*
named constructor for IVM-based BB
*/
#ifndef MAKE_IVM_BB_H
#define MAKE_IVM_BB_H

#include <memory>
#include <arguments.h>
#include <pbab.h>

#include "../base/mcbb.h"
#include "../ivm/intervalbb.h"
#include "set_operators.h"

template<typename T>
std::shared_ptr<Intervalbb<T>> make_ivmbb(pbab* pbb);

#endif
