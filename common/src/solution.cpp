#include "../include/pbab.h"
#include "../include/solution.h"
#include "../include/macros.h"

#include "../include/log.h"

#include <pthread.h>
#include <atomic>


solution::solution(int _size)
{
    size = _size;

    perm = (int *)calloc(size,sizeof(int));
    for(int i=0;i<size;i++)
    {
        perm[i]=i;
    }

    cost   = ATOMIC_VAR_INIT(INT_MAX);
    // std::atomic<int>{INT_MAX};
    // newBest    = false;

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
    pthread_mutex_init(&mutex_sol, &attr);

    // pthread_rwlock_init(&lock_sol,NULL);

}

int
solution::update(const int * candidate, const int _cost)
{
    int ret = 0;
    // pthread_mutex_lock(&mutex_sol);
    pthread_mutex_lock_check(&mutex_sol);
    if (_cost < cost.load()) {
        ret = 1;
        // newBest  = true;
        cost.store(_cost);
        if (candidate) {
            for (int i = 0; i < size; i++) perm[i] = candidate[i];
        }
    }
    pthread_mutex_unlock(&mutex_sol);
    return ret;
}

int
solution::updateCost(const int _cost)
{
    int ret = 0;
    pthread_mutex_lock_check(&mutex_sol);
    if (_cost < cost.load()) {
        ret = 1;
        cost.store(_cost);
    }
    pthread_mutex_unlock(&mutex_sol);
    return ret;
}

void
solution::getBestSolution(int *_perm, int &_cost)
{
    pthread_mutex_lock_check(&mutex_sol);
    _cost = cost.load();
    for (int i = 0; i < size; i++)
        _perm[i] = perm[i];
    pthread_mutex_unlock(&mutex_sol);
}

// https://stackoverflow.com/questions/16190078/how-to-atomically-update-a-maximum-value
template<typename T>
void update_maximum(std::atomic<T>& maximum_value, T const& value) noexcept
{
    T prev_value = maximum_value;
    while(prev_value < value &&
            !maximum_value.compare_exchange_weak(prev_value, value))
        {}
}

template<typename T>
void update_minimum(std::atomic<T>& minimum_value, T const& value) noexcept
{
    T prev_value = minimum_value;
    while(prev_value > value &&
            !minimum_value.compare_exchange_weak(prev_value, value))
        {}
}

int
solution::getBest()
{
    int ret;

    // pthread_rwlock_rdlock(&lock_sol);
    ret=cost.load();
    // pthread_rwlock_unlock(&lock_sol);
    return ret;
}

void
solution::getBest(int& _cost)
{
    // pthread_rwlock_rdlock(&lock_sol);
    if (cost.load() < _cost) {
        _cost = cost.load();
    }
    // pthread_rwlock_unlock(&lock_sol);
    return;
}

void
solution::print()
{
    pthread_mutex_lock_check(&print_mutex);
    std::cout<<cost.load()<<"\t|\t";
    for (int i = 0; i < size; i++) {
        std::cout<<perm[i]<<" ";
    }
    std::cout<<std::endl;
    pthread_mutex_unlock(&print_mutex);
}

void
solution::save()
{
    pthread_mutex_lock_check(&mutex_sol);
    FILE_LOG(logINFO) << "SAVE SOLUTION " << this->cost.load();

    std::ofstream stream(("./bbworks/sol" + std::string(arguments::inst_name) + ".save").c_str());
    stream << *this <<std::endl;
    stream.close();
    pthread_mutex_unlock(&mutex_sol);
}

solution&
solution::operator=(solution& s)
{
    size = s.size;

    cost.store(s.cost.load());
    // std::atomic_store(mInt, std::atomic_load(pOther.mInt, memory_order_relaxed), memory_order_relaxed);
    // cost   = s.cost;
    // newBest    = s.newBest;

    for(int i=0;i<size;i++)
    {
        perm[i]=s.perm[i];
    }
    return *this;
}

// write solution to stream
std::ostream&
operator << (std::ostream& stream, const solution& s)
{
    stream << s.size << std::endl;
    stream << s.cost << std::endl;
    for (int i = 0; i < s.size; i++) {
        stream << s.perm[i] << " ";
    }
    stream << std::endl;

    return stream;
}

// read solution from stream
std::istream&
operator >> (std::istream& stream, solution& s)
{
    stream >> s.size;

    int tmp;
    stream >> tmp;
    s.cost.store(tmp);
    for (int i = 0; i < s.size; i++) {
        stream >> s.perm[i];
    }
    return stream;
}
