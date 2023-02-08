/*
victim selection strategies

- ring (round-robin)
- random
- honest
*/
#ifndef VICTIM_SELECTOR_H
#define VICTIM_SELECTOR_H

#include <random>
#include <vector>
#include <memory>
#include <list>
#include <iostream>


class VictimSelector
{
public:
    VictimSelector(unsigned _nthreads) : nthreads(_nthreads){};

    virtual unsigned operator()(unsigned id) = 0;

protected:
    unsigned nthreads;
};

class RingVictimSelector : public VictimSelector
{
public:
    explicit RingVictimSelector(unsigned _nthreads) : VictimSelector(_nthreads){};

    unsigned operator()(unsigned id)
    {
        return (id == 0) ? (nthreads - 1) : (id - 1);
    }
};

//select victim randomly (may select idle victim)
class RandomVictimSelector : public VictimSelector
{
public:
    explicit RandomVictimSelector(unsigned _nthreads) : VictimSelector(_nthreads)
    {

        auto seed = static_cast<long int>(std::time(nullptr));
        random_engine = std::mt19937(size_t(seed));
        // generator = std::random_device{ } ();
        unif = std::uniform_int_distribution<int>(0,nthreads-1);
    };

    unsigned operator()(unsigned id)
    {
        unsigned victim = (id == 0) ? (nthreads - 1) : (id - 1);
        unsigned int attempts = 0;

        do {
            // randomly select active thread (at most nbIVM attempts...otherwise loop may be infinite)
            victim = rand() / (RAND_MAX /  nthreads);
            if(++attempts > nthreads){
                break;
            }
        }while(victim == id);

        return victim;
    }
private:
    std::mt19937 random_engine;
    std::uniform_int_distribution<int> unif;
};

class HonestVictimSelector : public VictimSelector
{
public:
    explicit HonestVictimSelector(unsigned _nthreads) : VictimSelector(_nthreads){
        for (unsigned i = 0; i < nthreads; i++) {
            victim_list.push_back(i);
        }

        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);

        pthread_mutex_init(&mutex_steal_list, &attr);
    };

    unsigned operator()(unsigned id)
    {
        unsigned victim = (id == 0) ? (nthreads - 1) : (id - 1);
        unsigned attempts = 0;

        pthread_mutex_lock(&mutex_steal_list);
        do{
            victim_list.remove(id);// remove id from list
            victim_list.push_back(id);// put at end

            victim = victim_list.front();// take first in list (oldest)
        }while(++attempts < nthreads);
        pthread_mutex_unlock(&mutex_steal_list);
        return victim;
    }

private:
    std::list<unsigned> victim_list;

    pthread_mutex_t mutex_steal_list;
};


std::unique_ptr<VictimSelector> make_victim_selector(const unsigned _nthreads, const char _type);

#endif
