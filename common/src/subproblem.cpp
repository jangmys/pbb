#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>

#include "../../common/include/misc.h"
#include "../include/subproblem.h"

subproblem::subproblem(int _size) :
    size(_size),limit2(size),
    schedule(std::vector<int>(size)),mask(std::vector<bool>(size,true))
{
    std::iota(schedule.begin(),schedule.end(),0);
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

    lb = s.lb;
	ub = s.ub;
	depth= s.depth;
}

subproblem::subproblem(const subproblem& father, const int indice, const int begin_end) : size(father.size),
    limit1(father.limit1 + 1 - begin_end),limit2(father.limit2 - begin_end),schedule(father.schedule),mask(std::vector<bool>(size,true)),depth(father.depth+1)
{
    if(begin_end == 0)
	{
        remove_insert_left(schedule.data(),limit1,indice);
	}else{
        remove_insert_right(schedule.data(),indice,limit2);
	}
}


void subproblem::swap(int a, int b)
{
    int tmp = schedule[a];
    schedule[a] = schedule[b];
    schedule[b] = tmp;
}

bool
subproblem::is_simple()  const
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
    printf("[");
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
    printf("]");
    printf("\t LB: %d",lb);
    printf("\t UB: %d\n",ub);
}

void
subproblem::shuffle()
{
    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(schedule.begin(), schedule.end(), g);
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

    schedule = s.schedule;
    mask = s.mask;

    depth = s.depth;
    prio = s.prio;
    lb = s.lb;
	ub = s.ub;

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
	stream << "\t" << s.lb << " " << s.ub;

    return stream;
}
