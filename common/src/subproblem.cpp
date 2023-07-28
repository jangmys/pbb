#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>

#include "../../common/include/misc.h"
#include "../include/subproblem.h"

subproblem::subproblem(int _size) :
    size(_size),limit1(-1),limit2(_size),
    schedule(std::vector<int>(size))
{
    std::iota(schedule.begin(),schedule.end(),0);

    lb = 0;
	_ub=0;
	depth=0;
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

    lb = s.lb;
	_ub = s._ub;
	depth= s.depth;
}

subproblem::subproblem(const subproblem& father, int indice, int begin_end)
{
    size = father.size;
    schedule = father.schedule;//std::vector<int>(size);

    limit1 = father.limit1 + 1 - begin_end;
    limit2 = father.limit2 - begin_end;

    if(begin_end == 0)
	{
        remove_insert_left(schedule.data(),limit1,indice);
	}else{
        remove_insert_right(schedule.data(),indice,limit2);
	}

    depth=father.depth+1;
    lb = 0;
    _ub=0;
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
    printf("\t %d\n",lb);
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

    for (int j = 0; j < size; j++)
        schedule[j] = s.schedule[j];

    lb = s.lb;
	_ub = s._ub;
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
	stream << "\t" << s.lb << " " << s._ub;

    return stream;
}
