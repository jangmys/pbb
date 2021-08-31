#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <random>
#include <numeric>

#include "../../common/include/misc.h"
#include "../include/subproblem.h"

#define FWDBR 0
#define BWDBR 1

subproblem::subproblem(int _size) :
    size(_size),limit1(-1),limit2(_size),
    schedule(std::vector<int>(size))
{
    // schedule = std::vector<int>(size);
    std::iota(schedule.begin(),schedule.end(),0);

    cost = 0;
	ub=0;
	depth=0;
	// prio = 0;

    // mask = std::vector<bool>(size);

    // for (int j = 0; j < size; j++) schedule[j] = j;
}

subproblem::~subproblem()
{
    // free(schedule);
}

subproblem::subproblem(const subproblem& s)
{
    size = s.size;
    schedule = std::vector<int>(size);
    // schedule = (int*)malloc(sizeof(int) * size);
    limit1 = s.limit1;
    limit2 = s.limit2;

    for (int j = 0; j < size; j++)
        schedule[j] = s.schedule[j];

    cost = s.cost;
	ub = s.ub;
	depth= s.depth;
	// prio = s.prio;
    // limites_set(father, begin_end);
    // permutation_set(father, indice, begin_end);
}

subproblem::subproblem(const subproblem& father, int indice, int begin_end)
{
    size = father.size;
    schedule = std::vector<int>(size);
    // schedule = (int*)malloc(sizeof(int) * size);

    limites_set(father, begin_end);
    permutation_set(father, indice, begin_end);

    depth=father.depth+1;
    cost = 0;
    ub=0;
}


int subproblem::locate(const int job)
{
    for(int i=0;i<size;i++)
    {
        if(schedule[i]==job)
            return i;
    }
    return -1; //error: not found
}

void subproblem::swap(int a, int b)
{
    int tmp = schedule[a];
    schedule[a] = schedule[b];
    schedule[b] = tmp;
}

void subproblem::removeInsertLeft(int a, int b)
{
    int tmp = schedule[b];

    for(int i=b;i>a;i--)
    {
        schedule[i] = schedule[i-1];
    }

    schedule[a] = tmp;
}

void subproblem::removeInsertRight(int a, int b)
{
    int tmp = schedule[a];

    for(int i=a;i<b;i++)
    {
        schedule[i] = schedule[i+1];
    }

    schedule[b] = tmp;
}





void
subproblem::limites_set(const subproblem& father, int begin_end)
{
    int a = (begin_end == FWDBR)?0:1;
    limit1 = father.limit1 + 1 - a;
    limit2 = father.limit2 - a;
}

void
subproblem::permutation_set(const subproblem& father, int indice, int begin_end)
{
    // job = father.schedule[indice];
    for (int j = 0; j < size; j++)
        schedule[j] = father.schedule[j];

    //swap
    // int tmp_indice = (begin_end == FWDBR) ? father.limit1 + 1 : father.limit2 - 1;
	// int tmp        = schedule[tmp_indice];
    // schedule[tmp_indice] = schedule[indice];
    // schedule[indice]     = tmp;

	//insert
	if(begin_end == FWDBR)
	{
		int tmp = schedule[indice];
		for(int j = indice; j > father.limit1+1; j--)
		{
			schedule[j]=schedule[j-1];
		}
		schedule[father.limit1+1]=tmp;
	}else{
		int tmp = schedule[indice];
		for(int j = indice; j < father.limit2-1; j++)
		{
			schedule[j]=schedule[j+1];
		}
		schedule[father.limit2-1]=tmp;
	}
}

bool
subproblem::simple()  const
{
    return (limit2-limit1 == 3);
}

bool
subproblem::leaf()  const
{
    return (limit2-limit1 == 2);
}

void
subproblem::print()
{
    for(int i=0;i<=limit1;i++)
    {
        printf("%3d ",schedule[i]);
    }
    printf(" | ");
    for(int i=limit1+1;i<limit2;i++)
    {
        printf("%3d ",schedule[i]);
    }
    printf(" | ");
    for(int i=limit2;i<size;i++)
    {
        printf("%3d ",schedule[i]);
    }
    printf("\t %d\n",cost);
}



// void
// subproblem::copy(subproblem *p)
// {
//     for(int i=0;i<size;i++){
//         schedule[i]=p->schedule[i];
//     }
//     limit1=p->limit1;
//     limit2=p->limit2;
// }

//for debugging

// int
// subproblem::intRand(const int & min, const int & max) {
//     static thread_local std::mt19937 generator;
//     std::uniform_int_distribution<int> distribution(min,max);//closed [min,max]
//     return distribution(generator);
// }

void
subproblem::shuffle()
{
    int n=size;
    if (n > 1) {
	    for (int i = 0; i < n - 1; i++) {
	        size_t j = helper::intRand(i,n-1);
			// printf("%d %d\n",j,n);
			// i + drand48() / (RAND_MAX / (n - i) + 1);
	        int t = schedule[j];
	        schedule[j] = schedule[i];
	        schedule[i] = t;
	    }
    }
}

subproblem&
subproblem::operator=(const subproblem& s)
{
    // self-assignment guard
    if (this == &s)
        return *this;

    size = s.size;
    limit1 = s.limit1;
    limit2 = s.limit2;

    for (int j = 0; j < size; j++)
        schedule[j] = s.schedule[j];

    cost = s.cost;
	ub = s.ub;
    depth = s.depth;

    return *this;
}

// write subproblem to stream
std::ostream&
operator << (std::ostream& stream, const subproblem& s)
{
	stream << s.limit1 << " ";
	stream << s.limit2 << "\t";
    for (int i = 0; i < s.size; i++) {
        stream << s.schedule[i] << " ";
    }
	stream << "\t" << s.cost << " " << s.ub;

    return stream;
}
