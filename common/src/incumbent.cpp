#include "../include/pbab.h"
#include "../include/incumbent.h"
#include "../include/macros.h"

#include "../include/log.h"

#include <pthread.h>
#include <atomic>

template<typename T>
Incumbent<T>::Incumbent(int _size)
{
    size = _size;

    perm = std::vector<int>(size,0);
    initial_perm = std::vector<int>(size,0);

    for(int i=0;i<size;i++)
    {
        perm[i]=i;
        initial_perm[i]=i;
    }

    cost   = ATOMIC_VAR_INIT(INT_MAX);
    initial_cost   = ATOMIC_VAR_INIT(INT_MAX);

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
    pthread_mutex_init(&mutex_sol, &attr);

    // pthread_rwlock_init(&lock_sol,NULL);

}

template<typename T>
int Incumbent<T>::update(const int * candidate, const int _cost)
{
    int ret = 0;
    // pthread_mutex_lock(&mutex_sol);
    pthread_mutex_lock_check(&mutex_sol);
    if (_cost <= cost.load()) {
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

template<typename T>
int Incumbent<T>::updateCost(const int _cost)
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

template<typename T>
void Incumbent<T>::getBestSolution(int *_perm, int &_cost)
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

template<typename T>
int Incumbent<T>::getBest()
{
    int ret;

    // pthread_rwlock_rdlock(&lock_sol);
    ret=cost.load(std::memory_order_relaxed);
    // pthread_rwlock_unlock(&lock_sol);
    return ret;
}

template<typename T>
void Incumbent<T>::getBest(int& _cost)
{
    // pthread_rwlock_rdlock(&lock_sol);
    if (cost.load() < _cost) {
        _cost = cost.load(std::memory_order_relaxed);
    }
    // pthread_rwlock_unlock(&lock_sol);
    return;
}

pthread_mutex_t print_incumbent_mutex = PTHREAD_MUTEX_INITIALIZER;

template<typename T>
void Incumbent<T>::print()
{
    pthread_mutex_lock_check(&print_incumbent_mutex);
    std::cout<<cost.load()<<",[";
    for (int i = 0; i < size; i++) {
        std::cout<<perm[i]<<" ";
    }
    std::cout<<"]"<<std::endl;
    pthread_mutex_unlock(&print_incumbent_mutex);
}

template<typename T>
void Incumbent<T>::save()
{
    pthread_mutex_lock_check(&mutex_sol);
    FILE_LOG(logINFO) << "SAVE SOLUTION " << this->cost.load() << " to " << ("./output/sol" + std::string(arguments::inst_name) + ".save");

    std::ofstream stream(("./output/sol" + std::string(arguments::inst_name) + ".save").c_str());
    stream << *this <<std::endl;
    stream.close();
    pthread_mutex_unlock(&mutex_sol);
}

template<typename T>
Incumbent<T>& Incumbent<T>::operator=(Incumbent<T>& s)
{
    size = s.size;

    cost.store(s.cost.load());

    for(int i=0;i<size;i++)
    {
        perm[i]=s.perm[i];
    }
    return *this;
}

template class Incumbent<int>;
