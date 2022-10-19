#ifndef FACT_WORK_H
#define FACT_WORK_H

#include <memory>
#include <vector>

#include "work.h"

#include "gmp.h"
#include "gmpxx.h"

class work;
class weights;

class fact_work{
public:
    int id;
    unsigned nb_intervals;
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


// namespace convert
// {
//     static std::shared_ptr<fact_work> dec2fact(std::shared_ptr<work> dw, int pbsize)
//     {
//         auto ret = std::make_shared<fact_work>(dw->max_intervals, pbsize);
//
//         ret->id = dw->id;
//         ret->nb_intervals  = dw->Uinterval.size();
//         ret->max_intervals = dw->max_intervals;
//         ret->nb_decomposed = dw->nb_decomposed;
//         ret->nb_leaves = dw->nb_leaves;
//
//         for (unsigned k = 0; k < ret->nb_intervals; k++) {
//             ret->ids[k] = dw->Uinterval[k]->id;
//             ret->BigintToVect(
//                 dw->Uinterval[k]->begin, dw->Uinterval[k]->end,
//                 ret->pos + k * pbsize, ret->end + k * pbsize
//             );
//         }
//         return ret;
//     }
//
//     static std::shared_ptr<work> fact2dec(std::shared_ptr<fact_work> fw, int pbsize)
//     {
//         auto ret = std::make_shared<work>();
//
//         ret->id = fw->id;
//         ret->nb_intervals  = fw->nb_intervals;
//         ret->max_intervals = fw->max_intervals;
//         ret->nb_decomposed = fw->nb_decomposed;
//         ret->nb_leaves = fw->nb_leaves;
//
//         ret->Uinterval.clear();
//
//         mpz_class tmpb(0);
//         mpz_class tmpe(0);
//
//         for (unsigned i = 0; i < fw->nb_intervals; i++) {
//             fw->VectToBigint(fw->pos + i * pbsize, fw->end + i * pbsize, tmpb, tmpe);
//             // std::cout<<"V2I\t"<<tmpb<<" "<<tmpe<<std::endl;
//
//             if (tmpb < tmpe) {
//                 ret->Uinterval.emplace_back(new interval(tmpb, tmpe, fw->ids[i]));
//             }
//         }
//
//         ret->sortIntervals();
//         return ret;
//         // dw->displayUinterval();
//     }
//
//
// }



#endif
