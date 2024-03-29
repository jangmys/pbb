#include <stdlib.h>
#include <string.h>

#include "log.h"

#include "work.h"
#include "fact_work.h"


// in case max is not known yet
fact_work::fact_work(int _size) :
    id(0),nb_intervals(0),max_intervals(0),nb_decomposed(0),nb_leaves(0),pbsize(_size),wghts(_size)
{}

fact_work::fact_work(int _max, int _size) : id(0),nb_intervals(0),max_intervals(_max),nb_decomposed(0),nb_leaves(0),pbsize(_size),wghts(_size)
{
    states = (int *) calloc(_max, sizeof(int));
    ids    = (int *) calloc(_max, sizeof(int));

    pos = (int *) calloc(_max * pbsize, sizeof(int));
    end = (int *) calloc(_max * pbsize, sizeof(int));

    FILE_LOG(logDEBUG1) << "Work-buffer, MaxIntervals " << _max << " Bytes: " << 2 * _max * (pbsize + 1) * sizeof(int);
}

void
fact_work::BigintToVect(mpz_class begin, int * posV)
{
    mpz_class q(0);
    mpz_class r(begin);

    for (int i = pbsize; i > 0; i--) {
        mpz_tdiv_qr(q.get_mpz_t(), r.get_mpz_t(), r.get_mpz_t(), wghts.depth[pbsize - i + 1].get_mpz_t());
        posV[pbsize - i] = q.get_ui();
    }
    // r = end;
    // for (int i = pbsize; i > 0; i--) {
    //     mpz_tdiv_qr(q.get_mpz_t(), r.get_mpz_t(), r.get_mpz_t(), wghts.depth[pbsize - i + 1].get_mpz_t());
    //     endV[pbsize - i] = q.get_ui();
    // }
}

void
fact_work::VectToBigint(const int * posV, const int * endV, mpz_class &begin, mpz_class &end)
{
    begin = 0;
    end   = 0;

    for (int i = pbsize - 1; i >= 0; i--) {
        begin += wghts.W[i + 1][posV[i]];
        end   += wghts.W[i + 1][endV[i]];
    }
}

void
fact_work::fact2dec(std::shared_ptr<work> dw)
{
    dw->id = this->id;
    dw->nb_intervals  = nb_intervals;
    dw->max_intervals = max_intervals;
    dw->nb_decomposed = nb_decomposed;
    dw->nb_leaves = nb_leaves;

    dw->Uinterval.clear();

    mpz_class tmpb(0);
    mpz_class tmpe(0);

    for (unsigned i = 0; i < nb_intervals; i++) {
        // for (int j = 0; j < pbsize; j++) {
        //     printf("%d ", pos[i*pbsize+j]);
        // }
        // printf("\n");
        // for (int j = 0; j < pbsize; j++) {
        //     printf("%d ", end[i*pbsize+j]);
        // }
        // printf("\n");

        VectToBigint(pos + i * pbsize, end + i * pbsize, tmpb, tmpe);
        // std::cout<<"V2I\t"<<tmpb<<" "<<tmpe<<std::endl;

        if (tmpb < tmpe) {
            dw->Uinterval.emplace_back(std::make_shared<interval>(tmpb, tmpe, ids[i]));
        }
    }

    dw->sortIntervals();
    // dw->displayUinterval();
}

void
fact_work::dec2fact(std::shared_ptr<work> dw)
{
    id = dw->id;
    nb_intervals  = dw->Uinterval.size();
    max_intervals = dw->max_intervals;
    nb_decomposed = dw->nb_decomposed;
    nb_leaves = dw->nb_leaves;

    for (unsigned k = 0; k < nb_intervals; k++) {
        ids[k] = dw->Uinterval[k]->id;
        BigintToVect(dw->Uinterval[k]->begin, pos + k * pbsize);
        BigintToVect(dw->Uinterval[k]->end, end + k * pbsize);
    }
}
