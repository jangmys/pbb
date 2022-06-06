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

    void
    setRoot(const int* root, int l1, int l2);

    void push(std::unique_ptr<subproblem>& p);


    int size();
    void push_back(std::shared_ptr<subproblem> p);
    void pop();
    std::shared_ptr<subproblem> top();
    bool empty();
    std::shared_ptr<subproblem> take();
    void clearPool();

    void insert(std::vector<std::shared_ptr<subproblem>>& ns);
};


#endif
