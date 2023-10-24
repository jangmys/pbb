#ifndef SUBPROBLEM_H
#define SUBPROBLEM_H

#include <vector>
#include <iostream>


class subproblem {
    friend std::ostream& operator << (std::ostream& stream, const subproblem& s);

public:
    int size = 0;
    int limit1 = -1;
    int limit2;

    std::vector<int> schedule;
    std::vector<bool> mask;

    int depth = 0;
    float prio = 0 ;
    int lb = 0;
    int ub = 0;

    subproblem(int _size);
    subproblem(const int _size,const std::vector<int> _arr) : size(_size),limit2(size),schedule(_arr),mask(std::vector<bool>(size,true)){};

    subproblem(const subproblem& s);
    subproblem(const subproblem& father, const int indice, const int begin_end);

    ~subproblem() = default;

    subproblem&
    operator = (const subproblem& s);


    bool
    is_simple()  const;
    bool
    leaf()  const;

    void
    print();
    void
    shuffle();

    void swap(int a, int b);
};

std::ostream&
operator << (std::ostream& stream, const subproblem& s);

#endif // ifndef SUBPROBLEM_H
