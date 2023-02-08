#ifndef POOL_H_
#define POOL_H_

#define DEQUE       'd'
#define STACK       's'
#define PRIOQ       'p'
#define VECTOR      'v'

#include <vector>
#include <stack>
#include <queue>
#include <memory>

#include <subproblem.h>

/*
basically a proxy for stack, deque, priority_queue, etc
*/
class Pool
{
private:
    int psize;

public:
    Pool(int _size);

    int strategy;

    std::vector<std::unique_ptr<subproblem>> activeSet;
    std::deque<std::unique_ptr<subproblem>> deq;
    std::stack<std::unique_ptr<subproblem>> pile;
    // std::priority_queue<std::shared_ptr<subproblem>, std::vector<std::shared_ptr<subproblem>>, ub_compare> pque;

    void insert(const int* root, int l1, int l2, unsigned int psize);
    void insert(std::unique_ptr<subproblem> p);

    std::unique_ptr<subproblem> top();
    std::unique_ptr<subproblem> back();
    std::unique_ptr<subproblem> take_top(); //top + pop
    std::unique_ptr<subproblem> take_back(); //back + pop_back

    void pop();
    bool empty();
    void pop_back();

    int size();
};


#endif
