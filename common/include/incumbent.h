#ifndef INCUMBENT_H
#define INCUMBENT_H

#include <atomic>
#include <vector>
#include <iostream>
#include <climits>
#include <pthread.h>

/*
 * this class and subproblem are somewhat redundant....
 */
template<typename T>
class Incumbent {
public:
    Incumbent(int _size);

    int size;

    std::atomic<T> initial_cost;
    std::vector<int> initial_perm;

    std::atomic<T>cost;
    std::vector<int> perm;

    std::atomic<bool> foundAtLeastOneSolution{false};
    std::atomic<bool> foundNewSolution{false};

    pthread_mutex_t mutex_sol;
    pthread_rwlock_t lock_sol;

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

    Incumbent&
    operator = (Incumbent& s);

    friend std::ostream& operator << (std::ostream& stream, const Incumbent& s)
    {
        stream << s.size << std::endl;
        stream << s.cost << std::endl;
        for (int i = 0; i < s.size; i++) {
            stream << s.perm[i] << " ";
        }
        stream << std::endl;

        return stream;
    };

    friend std::istream& operator >> (std::istream& stream, Incumbent& s){
        stream >> s.size;

        int tmp;
        stream >> tmp;
        s.cost.store(tmp);
        for (int i = 0; i < s.size; i++) {
            stream >> s.perm[i];
        }
        return stream;
    };
};

// std::ostream&
// operator << (std::ostream& stream, const Incumbent& s);
// std::istream&
// operator >> (std::istream& stream, Incumbent& s);
#endif // ifndef SOLUTION_H
