#ifndef WORK_H
#define WORK_H

//=====================================================
#include <pthread.h>
#include <iostream>
#include <vector>
#include <queue>
#include <memory>

#include "interval.h"

#include "gmp.h"
#include "gmpxx.h"

// class pbab;
class interval;

typedef std::shared_ptr<interval> INTERVAL_PTR;
typedef std::vector<std::shared_ptr<interval>> INTERVAL_VEC;
typedef std::vector<std::shared_ptr<interval>>::iterator INTERVAL_IT;

class work
{
public:
    INTERVAL_VEC Uinterval; //...as decimals

    //the meta-data
    int id;//work ID
    int nb_intervals;
    int max_intervals;
    int exploredNodes;
    int nbLeaves;

    int end_updated;//flag : was end modified?
    int nb_updates;

    mpz_class size;

    //Constructeurs
    work();
    work(const work &w);
    work(std::istream& stream);

    ~work();

    size_t writeToFile(FILE*);
    size_t readFromFile(FILE*);

    void readFromBuffer(int* buffer);

    void sortIntervals();
    void sortBest();

    void displayUinterval();
    void clear();
    mpz_class wsize();

    //work operators
    void  unionn(work* w);
//    void intersection(const std::shared_ptr<work>& w, bool&);
    bool intersection(const std::shared_ptr<work>& w);


    void subtractFromAll(std::shared_ptr<work> w);
    work* subtraction(work* w);
    std::shared_ptr<work> divide(int max);
    std::shared_ptr<work> divide2(int max);
    std::shared_ptr<work> take(int max);

    void streamToWork(std::string s);

    void readHeader(std::istream& stream);
    int readIntervals(std::istream& stream);
    void writeHeader(std::ostream& stream)const;
    void writeIntervals(std::ostream& stream)const;

    //Gestion des variables membres
    void set_size();
    void set_time();
    // void set_peer(peer& p);
    void set_finished();
    void set_null();
    void set_id();

 //checking
 // bool big();
 // bool big2();
 // bool finished();
 bool isEmpty();
 bool fault();
 bool update();

 bool disjoint(work* w);
 bool contain(work* w);

 //operators
    void operator=(work& w);
    bool equal(work& w);
    bool differ(std::shared_ptr<work>& w);

    void VectToBigint(const int *posV,const int *endV, mpz_class &begin, mpz_class &end);
    void BigintToVect(mpz_class begin,mpz_class end, int *posV, int *endV);

    //new...
    int renumber();
    void split(size_t n);
    void split2(size_t n);

};

std::ostream&  operator<<(std::ostream& stream,const work& w);
std::istream& operator>>(std::istream& stream, work& w);
void work_free(work*);
#endif
