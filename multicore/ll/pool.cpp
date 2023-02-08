#include "pool.h"


Pool::Pool(int _size)
{
    //data structure used for pool of subproblems (default)
    strategy = DEQUE;
}

//========> cf IVM
void
Pool::insert(const int* perm, int l1, int l2, unsigned int psize)
{
    std::unique_ptr<subproblem> root = std::make_unique<subproblem>(psize);

    for(int i=0;i<psize;i++){
        root->schedule[i]=perm[i];
    }
    root->set_lower_bound(0);
    root->limit1=l1;
    root->limit2=l2;

    insert(std::move(root));
}

void
Pool::insert(std::unique_ptr<subproblem> p)
{
    switch (strategy) {
        case DEQUE:
            deq.push_front(std::move(p));
            break;
        case STACK:
            pile.push(std::move(p));
            break;
        // case PRIOQ:
        //     pque.push(p);
        //     break;
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}

std::unique_ptr<subproblem>
Pool::take_top()
{
    std::unique_ptr<subproblem> n=(empty()) ? NULL : std::move(top());
    if(n) pop();

    return std::move(n);
}

std::unique_ptr<subproblem>
Pool::take_back()
{
    std::unique_ptr<subproblem> n=(empty()) ? NULL : std::move(back());
    if(n) pop_back();

    return std::move(n);
}

std::unique_ptr<subproblem>
Pool::top()
{
    switch (strategy) {
        case DEQUE:
            return std::move(deq.front());
        case STACK:
            return std::move(pile.top());
        // case PRIOQ:
        //     return pque.top();
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}

std::unique_ptr<subproblem>
Pool::back()
{
    switch (strategy) {
        case DEQUE:
            return std::move(deq.back());
        case STACK:
            return std::move(pile.top());
        // case PRIOQ:
        //     pque.pop();
        //     break;
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}

void
Pool::pop()
{
    switch (strategy) {
        case DEQUE:
            deq.pop_front();
            break;
        case STACK:
            pile.pop();
            break;
        // case PRIOQ:
        //     pque.pop();
        //     break;
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}

void
Pool::pop_back()
{
    switch (strategy) {
        case DEQUE:
            deq.pop_back();
            break;
        case STACK:
            pile.pop();
            break;
        // case PRIOQ:
        //     pque.pop();
        //     break;
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}


bool
Pool::empty()
{
    switch (strategy){
        case DEQUE:
            return deq.empty();
        case STACK:
            return pile.empty();
        // case PRIOQ:
        //     return pque.empty();
        default:
            std::cout << "Undefined strategy";
            exit(1);
    }
}

int
Pool::size()
{
    switch (strategy) {
        case DEQUE:
            return deq.size();
        case STACK:
            return pile.size();
        // case PRIOQ:
        //     return pque.size();
        default:
            std::cout << "Undefined DS";
            exit(1);
    }
}
