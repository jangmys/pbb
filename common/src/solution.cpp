#include "../include/pbab.h"
#include "../include/solution.h"
#include "../include/macros.h"

#include "../include/log.h"



solution::solution(int _size)
{
    size = _size;

    perm = (int *)calloc(size,sizeof(int));
    for(int i=0;i<size;i++)
    {
        perm[i]=i;
    }

    cost   = INT_MAX;
    // newBest    = false;

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
    pthread_mutex_init(&mutex_sol, &attr);
}

int
solution::update(const int * candidate, const int _cost)
{
    int ret = 0;
    // pthread_mutex_lock(&mutex_sol);
    pthread_mutex_lock_check(&mutex_sol);
    if (_cost < cost) {
        ret = 1;
        // newBest  = true;
        cost = _cost;
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
    if (_cost < cost) {
        ret = 1;
        cost = _cost;
    }
    pthread_mutex_unlock(&mutex_sol);
    return ret;
}

void
solution::getBestSolution(int *_perm, int &_cost)
{
    pthread_mutex_lock_check(&mutex_sol);
    _cost = cost;
    for (int i = 0; i < size; i++)
        _perm[i] = perm[i];
    pthread_mutex_unlock(&mutex_sol);
}

int
solution::getBest()
{
    int ret;
    pthread_mutex_lock_check(&mutex_sol);
    ret=cost;
    pthread_mutex_unlock(&mutex_sol);
    return ret;
}

//no lock?? reading should be ok but better be safe....
void
solution::getBest(int& _cost)
{
    // if (cost == _cost) return;

    // pthread_mutex_lock_check(&mutex_sol);
    if (cost < _cost) {
        _cost = cost;
    }
    // pthread_mutex_unlock(&mutex_sol);

    return;
}

void
solution::print()
{
    pthread_mutex_lock_check(&print_mutex);
    std::cout<<"Cost: "<<cost<<"\t";
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
    FILE_LOG(logINFO) << "SAVE SOLUTION " << this->cost;

    std::ofstream stream(("./bbworks/sol" + std::string(arguments::inst_name) + ".save").c_str());
    stream << *this <<std::endl;
    stream.close();
    pthread_mutex_unlock(&mutex_sol);
}

solution&
solution::operator=(solution& s)
{
    size = s.size;

    cost   = s.cost;
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
    stream >> s.cost;
    for (int i = 0; i < s.size; i++) {
        stream >> s.perm[i];
    }
    return stream;
}
