#ifndef SINGLE_POOL_H
#define SINGLE_POOL_H

#include <stack>
#include <memory>
#include <vector>
#include <random>

#include <base_subproblem.h>

template<class Subproblem>
class SinglePool
{
public:
    SinglePool(){    };

    std::deque<std::unique_ptr<Subproblem>>deque;

    std::unique_ptr<Subproblem> take(){
        std::unique_ptr<Subproblem> n=(deque.empty())?nullptr:std::move(deque.front());
        if(n)deque.pop_front();
        return n;
    };

    std::unique_ptr<Subproblem> take_back(){
        std::unique_ptr<Subproblem> n=(deque.empty())?nullptr:std::move(deque.back());
        if(n)deque.pop_back();
        return n;
    };


    void insert(std::unique_ptr<Subproblem> n){
        deque.push_front(std::move(n));
    };

    void insert(std::vector<std::unique_ptr<Subproblem>> ns){
        for(auto &n : ns){
            deque.push_front(std::move(n));
        }
    };

    size_t size(){
        return deque.size();
    };

    bool empty()
    {
        return deque.empty();
    }
};

#endif
