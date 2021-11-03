#ifndef SUBPROBLEM_H
#define SUBPROBLEM_H

#include <vector>
#include <iostream>

class subproblem {
public:
    subproblem(int _size);
    subproblem(const subproblem& s);
    subproblem(const subproblem& father, int indice, int begin_end);

    ~subproblem();

    subproblem&
    operator = (const subproblem& s);

    int size;
    int limit1;
    int limit2;

    std::vector<int> schedule;
    std::vector<bool> mask;

    int cost;
    int ub;

    float prio;

    int depth;

    int
    locate(const int job);

    bool
    simple()  const;
    bool
    leaf()  const;

    // int intRand(const int & min, const int & max);

    void
    print();
    void
    shuffle();

    void swap(int a, int b);
    void removeInsertLeft(int a, int b);
    void removeInsertRight(int a, int b);

    void
    permutation_set(const subproblem& father, int indice, int begin_end);
    void
    limites_set(const subproblem& father, int begin_end);
    void
    copy(subproblem * p);

};

std::ostream&
operator << (std::ostream& stream, const subproblem& s);
// std::istream& operator>>(std::istream& stream, sumproblem& s);

#endif // ifndef SUBPROBLEM_H
