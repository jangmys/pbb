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
    SinglePool(){
        // omp_init_lock(&lock);
    };

    std::deque<std::unique_ptr<Subproblem>>deque;

    std::unique_ptr<Subproblem> take(){
        // omp_set_lock(&lock);
        std::unique_ptr<Subproblem> n=(deque.empty())?nullptr:std::move(deque.front());
        if(n)deque.pop_front();
        // omp_unset_lock(&lock);
        return n;
    };

    std::unique_ptr<Subproblem> take_back(){
        // omp_set_lock(&lock);
        std::unique_ptr<Subproblem> n=(deque.empty())?nullptr:std::move(deque.back());
        if(n)deque.pop_back();
        // omp_unset_lock(&lock);
        return n;
    };


    void insert(std::unique_ptr<Subproblem> n){
        // omp_set_lock(&lock);
        deque.push_front(std::move(n));
        // omp_unset_lock(&lock);
    };

    void insert(std::vector<std::unique_ptr<Subproblem>> ns){
        // omp_set_lock(&lock);
        for(auto &n : ns){
            // nnodes++;
            deque.push_front(std::move(n));
            // insert(n);
        }
        // omp_unset_lock(&lock);
    };

    size_t size(){
        return deque.size();
    };

    bool empty()
    {
        return deque.empty();
    }

private:
    // omp_lock_t lock;
};


//
//
// template<class Subproblem>
// class SharedPool
// {
// public:
//     SharedPool(unsigned _num_threads):segments(std::vector<Segment<Subproblem>>(_num_threads)),gen(rd()),distrib(0,_num_threads-1)
//     {
//         std::cout<<"num segments : "<<segments.size()<<"\n";
//     }
//
//     // std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
//     // std::uniform_int_distribution<> distrib(1, 6);
//
//
//     void insert(std::unique_ptr<Subproblem> n, const int tid){
//         segments[tid].insert(std::move(n));
//     }
//
//     void insert(std::vector<std::unique_ptr<Subproblem>> ns, const int tid){
//         segments[tid].insert(std::move(ns));
//     }
//
//     std::unique_ptr<Subproblem> take(const int tid){
//         std::unique_ptr<Subproblem> ret;
//
//         //best case : got work myself
//         ret = std::move(segments[tid].take());
//
//         //avg case : try get from neighbor
//         if(!ret){
//             int victim = distrib(gen);
//             // std::cout<<distrib(gen)<<'\n';
//
//             // int victim = tid?(tid-1):segments.size()-1;
//
//             ret = std::move(segments[victim].take_back());
//
//             // std::cout<<"thread "<<tid<<" got work from neighbor\n";
//
//             // if(!ret)
//             //     std::cout<<tid<<" get null\n";
//         }
//
//         return ret;
//     }
//
//     size_t size(const int tid){
//         return segments[tid].size();
//     };
//
//     bool empty(const int tid){
//         return segments[tid].empty();
//     }
//     // std::unique_ptr<Subproblem> take(){
//     //     omp_set_lock(&lock);
//     //     std::unique_ptr<Subproblem> n=(stack.empty())?nullptr:std::move(stack.top());
//     //     if(n)stack.pop();
//     //     omp_unset_lock(&lock);
//     //     return n;
//     // };
//
//
// private:
//     std::vector<Segment<Subproblem>> segments;
//
//     std::random_device rd;  //Will be used to obtain a seed for the random number engine
//     std::mt19937 gen;
//     std::uniform_int_distribution<> distrib;
//
//
//     // bool remove(Subproblem& s, const int tid){
//     //     if(segments[tid].empty()){
//     //         return false;
//     //     }else{
//     //         s=segments[tid].remove();
//     //         return true;
//     //     }
//     // }
//     //
//     // bool empty(){
//     //     bool ret;
//     //     omp_set_lock(&lock);
//     //     ret = stack.empty();
//     //     omp_unset_lock(&lock);
//     //     return ret;
//     // };
//     //
//     // size_t size(){
//     //     return nnodes;
//     // };
//     //
//     // void nnodes_decr_one(){
//     //     omp_set_lock(&lock);
//     //     nnodes--;
//     //     omp_unset_lock(&lock);
//     // };
//     //
//     // std::unique_ptr<Subproblem> top(){
//     //     std::unique_ptr<Subproblem> ret;
//     //
//     //     omp_set_lock(&lock);
//     //     ret = std::move(stack.top());
//     //     omp_unset_lock(&lock);
//     //
//     //     return std::move(ret);
//     // };
//     //
//     // void pop(){
//     //     omp_set_lock(&lock);
//     //     stack.pop();
//     //     omp_unset_lock(&lock);
//     // };
//     //
//     //
//
//     //
//     // void insert(std::vector<std::unique_ptr<Subproblem>> ns){
//     //     omp_set_lock(&lock);
//     //     for(auto &n : ns){
//     //         nnodes++;
//     //         stack.push(std::move(n));
//     //         // insert(n);
//     //     }
//     //     omp_unset_lock(&lock);
//     // };
//
// // private:
//
//     // unsigned long nnodes;
//     //
//     //
//     // std::stack<std::unique_ptr<Subproblem>>stack;
//     //
//     // omp_lock_t lock;
//
// };

#endif
