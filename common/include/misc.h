#ifndef MISC_H_
#define MISC_H_

#include <stdio.h>
#include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif

int negative(const int i);

int absolute(const int i);

void int_swap(int* a, int* b);

void gnomeSortByKeyInc(int * arr, const int * key, const int from, const int to);

void gnomeSortByKeysInc(int * arr, const int * key1, const int * key2, const int from, const int to);

#ifdef __cplusplus
}
#endif

#endif // ifndef MISC_H_
