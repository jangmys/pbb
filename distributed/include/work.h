/*
====================================================
A work unit (set of intervals)
--------------------------------------------------------
Author : Jan Gmys (jan.gmys@univ-lille.fr)
------------------------------------------
*/
#ifndef WORK_H
#define WORK_H

#include <pthread.h>
#include <iostream>
#include <vector>
#include <queue>
#include <memory>

#include "interval.h"

#include "gmp.h"
#include "gmpxx.h"

typedef std::shared_ptr<interval> INTERVAL_PTR;
typedef std::vector<std::shared_ptr<interval>> INTERVAL_VEC;
typedef std::vector<std::shared_ptr<interval>>::iterator INTERVAL_IT;

/*
A work unit for distributed PBB.

A collection of integer (arbitrary length) intervals.
*/
class work
{
public:
    INTERVAL_VEC Uinterval; //...as decimals

    //the meta-data
    int id = 0;//work ID
    int nb_intervals = 0;
    int max_intervals = 0;
    int nb_decomposed = 0;
    int nb_leaves = 0;

    int end_updated = 0;//flag : was end modified?
    int nb_updates = 0;

    mpz_class size = 0;

    //Constructeurs
    work();
    work(const work &w);
    work(std::istream& stream);
    ~work();

    //I/O
    size_t writeToFile(FILE*);
    size_t readFromFile(FILE*);
    void readFromBuffer(int* buffer);
    void readHeader(std::istream& stream);
    int readIntervals(std::istream& stream);
    void writeHeader(std::ostream& stream)const;
    void writeIntervals(std::ostream& stream)const;
    void displayUinterval();

    //Sort
    void sortIntervals();
    void sortBest();

    void clear();
    mpz_class wsize();

    //work operators
    bool intersection(const std::shared_ptr<work>& w);

    // work intersection(const std::shared_ptr<work>& w);

    std::shared_ptr<work> divide(int max);
    std::shared_ptr<work> take(int max);


    void set_size();
    void set_time();
    void set_finished();
    void set_null();
    void set_id();

    bool isEmpty();
    bool fault();
    bool update();

    bool disjoint(work* w);
    bool contain(work* w);

    //operators
    void operator=(work& w);
    bool equal(work& w);
    bool differ(std::shared_ptr<work>& w);

    //new...
    void split(size_t n);
    void split2(size_t n);
};

std::ostream&  operator<<(std::ostream& stream,const work& w);
std::istream& operator>>(std::istream& stream, work& w);

#endif
