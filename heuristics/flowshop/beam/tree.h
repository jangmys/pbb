#ifndef TREE_H
#define TREE_H

#define DEQUE       'd'
#define STACK       's'
#define PRIOQ       'p'

#define BEGIN_ORDER 0
#define END_ORDER   1

#define SIMPLE      0
#define JOHNSON     1

#include "../../common/inih/INIReader.h"

#include "libheuristic.h"
#include "libbounds.h"
#include "subproblem.h"

#include <semaphore.h>
#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <stack>
#include <queue>
#include <atomic>

class subproblem;
class pbab;

struct lb_compare {
    bool
    operator () (subproblem const * p1, subproblem const * p2) const
    {
        if(p1->cost > p2->cost)return true;
        if(p1->cost < p2->cost)return false;
        return (p1->depth < p2->depth);
        // return p1->cost < p2->cost;
    }
};

//p1 "greater" p2 --> smallest on top
struct ub_compare {
    bool
    operator () (subproblem const * p1, subproblem const * p2) const
    {
        //smaller cost first
        if(p1->ub > p2->ub)return true;
        if(p1->ub < p2->ub)return false;
        //depth first
        if(p1->depth < p2->depth)return true;
        if(p1->depth > p2->depth)return false;
        //smaller (weighted) idle time
        if(p1->prio > p2->prio)return true;
        if(p1->prio < p2->prio)return false;
        //smaller bound first
        if(p1->cost > p2->cost)return true;
        if(p1->cost < p2->cost)return false;

        return false;
    }
};

class Tree
{
private:
    int psize;

public:
    Tree(instance_abstract* inst, int _size);
    ~Tree();

    int strategy;

    // gestion pool
    std::deque<subproblem *> deq;
    std::stack<subproblem *> pile;
    std::priority_queue<subproblem *, std::vector<subproblem *>, ub_compare> pque;

    void
    setRoot(std::vector<int> root, int l1, int l2);

    int size();
    void push_back(subproblem * p);
    void push(subproblem * p);
    void pop();
    subproblem* top();
    bool empty();
    subproblem* take();
    void clearPool();

    void insert(std::vector<subproblem*>& ns);
};

#endif // ifndef TREE_H
