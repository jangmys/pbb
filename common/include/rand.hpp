#ifndef RAND_H_
#define RAND_H_

#include <random>
#include <algorithm>

// in closed interval [min,max]
int intRand(const int min, const int max);

float floatRand(const float min, const float max);
void shuffle(int * array, size_t n);

#endif
