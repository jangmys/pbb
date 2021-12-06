#ifndef FACT_WORK_H
#define FACT_WORK_H

#include <memory>
#include <vector>

#include "gmp.h"
#include "gmpxx.h"

class work;
class weights;

class fact_work{
public:
    int id;
    int nb_intervals;
    int max_intervals;
    int nb_decomposed;//decomposed nodes since last update
    int nb_leaves;

    int *ids;
    int *pos;
    int *end;

    int pbsize;
    int *states;

    weights* wghts;

    fact_work(int _size);
    fact_work(int _max, int _size);

    void BigintToVect(mpz_class begin, mpz_class end, int * posV, int * endV);
    void VectToBigint(const int * posV, const int * endV, mpz_class &begin, mpz_class &end);

    void fact2dec(std::shared_ptr<work>);
    void dec2fact(std::shared_ptr<work>);

    // void initAtNFact(int N);
    // void gather(std::vector<std::shared_ptr<fact_work>> fworks,const int best);
};

#endif
