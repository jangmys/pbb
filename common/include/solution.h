#ifndef SOLUTION_H
#define SOLUTION_H

#include <atomic>
#include <iostream>
#include <climits>
#include <pthread.h>



/*
 * this class and subproblem are somewhat redundant....
 */

class pbab;

class solution {
public:
    solution(int _size);

    int size;

    std::atomic<int>cost;
    // int cost;
    int * perm;

    pthread_mutex_t mutex_sol;

    pthread_rwlock_t lock_sol;
    // volatile bool newBest;

    // bool isImproved();


    int
    update(const int * candidate, const int cost);
    int
    updateCost(const int cost);

    void
    getBestSolution(int * permut, int &cost);

    int
    getBest();
    void
    getBest(int& cost);
    void
    print();

    void
    save();
    void
    random();

    solution&
    operator = (solution& s);
};

std::ostream&
operator << (std::ostream& stream, const solution& s);
std::istream&
operator >> (std::istream& stream, solution& s);
#endif // ifndef SOLUTION_H
