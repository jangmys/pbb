/*
=======================================
Work units in factoradic format (int*)
---------------------------------------
*/
#ifndef FACT_WORK_H
#define FACT_WORK_H

#include <memory>
#include <vector>

#include "work.h"

#include "gmp.h"
#include "gmpxx.h"

//Max number of jobs (max length of permutation)
constexpr size_t MAX_JOBS=800;

//// TODO : move weights and decimal stuff to work ?

/*
(N+1)*(N+1) table storing values

i*(j!) for i=0,1,...,N ; for j=0,1,...,N

Comes in handy when converting factoradic numbers to decimal and vice-versa
*/
class weights
{
public:
    //0 (N-1)!  2*(N-1)!    3*(N-1)!    ... (N-1)*(N-1)! N*(N-1)!
    //...
    //24
    //6
    //2
    //1
    //0 1   2   3   4   5   ... N-1 N
    weights(int _size)
    {
        depth[_size]     = 1;
        depth[_size - 1] = 1;
        for (int i = _size - 2, j = 2; i >= 0; i--, j++) {
            depth[i]  = j*depth[i + 1];
        }

        for (int i = 0; i <= _size; i++) {
            for (int j = 0; j <= _size; j++) {
                W[i][j] = j * depth[i];
            }
        }
    }

    mpz_class depth[MAX_JOBS+1];
    mpz_class W[MAX_JOBS+1][MAX_JOBS+1];
};



class fact_work{
public:
    int id;
    unsigned nb_intervals;
    int max_intervals;
    int nb_decomposed;//decomposed nodes since last update
    int nb_leaves;
    int pbsize;

    int *ids;
    int *pos;
    int *end;

    int *states;

    fact_work(int _size);
    fact_work(int _max, int _size);

    // void BigintToVect(mpz_class begin, mpz_class end, int * posV, int * endV);
    void BigintToVect(mpz_class begin, int * posV);
    void VectToBigint(const int * posV, const int * endV, mpz_class &begin, mpz_class &end);

    void fact2dec(std::shared_ptr<work>);
    void dec2fact(std::shared_ptr<work>);
private:
    weights wghts;
};

#endif
