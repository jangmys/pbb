#ifndef TREE_H
#define TREE_H

#define DEQUE       'd'
#define STACK       's'
#define PRIOQ       'p'
#define VECTOR      'v'

#define BEGIN_ORDER 0
#define END_ORDER   1

#define SIMPLE      0
#define JOHNSON     1

#include <memory>

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
    operator () (std::shared_ptr<subproblem> p1, std::shared_ptr<subproblem> p2) const
    {
        if(p1->lower_bound() > p2->lower_bound())return true;
        if(p1->lower_bound() < p2->lower_bound())return false;
        return (p1->depth < p2->depth);
    }
};

//p1 "greater" p2 --> smallest on top
struct ub_compare {
    bool
    operator () (std::shared_ptr<subproblem> p1, std::shared_ptr<subproblem> p2) const
    {
        //smaller cost first
        if(p1->fitness() > p2->fitness())return true;
        if(p1->fitness() < p2->fitness())return false;
        //depth first
        if(p1->depth < p2->depth)return true;
        if(p1->depth > p2->depth)return false;
        // //smaller (weighted) idle time
        // if(p1->prio > p2->prio)return true;
        // if(p1->prio < p2->prio)return false;
        //smaller bound first
        if(p1->lower_bound() > p2->lower_bound())return true;
        if(p1->lower_bound() < p2->lower_bound())return false;

        return false;
    }
};

class Tree
{
private:
    int psize;

public:
    Tree(instance_abstract* inst, int _size);

    int strategy;

    // gestion pool
    std::vector<std::shared_ptr<subproblem>> activeSet;
    std::deque<std::shared_ptr<subproblem>> deq;
    std::stack<std::shared_ptr<subproblem>> pile;
    std::priority_queue<std::shared_ptr<subproblem>, std::vector<std::shared_ptr<subproblem>>, ub_compare> pque;

    void
    setRoot(std::vector<int> root, int l1, int l2);

    int size();
    void push_back(std::shared_ptr<subproblem> p);
    void push(std::shared_ptr<subproblem> p);
    void pop();
    std::shared_ptr<subproblem> top();
    bool empty();
    std::shared_ptr<subproblem> take();
    void clearPool();

    void insert(std::vector<std::shared_ptr<subproblem>>& ns);
};

#endif // ifndef TREE_H
