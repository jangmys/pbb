#include <stdlib.h>

#ifndef C_BOUND_SIMPLE_H_
#define C_BOUND_SIMPLE_H_

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#ifdef __cplusplus
extern "C" {
#endif

//regroup (constant) bound data
typedef struct bound_data
{
    int *p_times;
    int *min_heads;    // for each machine k, minimum time between t=0 and start of any job
    int *min_tails;    // for each machine k, minimum time between release of any job and end of processing on the last machine
    int nb_jobs;
    int nb_machines;
} bound_data;

bound_data* new_bound_data(int _jobs, int _machines);
void free_bound_data(bound_data* b);


void add_forward(const int job, const int * const p_times, const int nb_jobs, const int nb_machines, int * front);

void add_backward(const int job, const int * const p_times, const int nb_jobs, const int nb_machines, int * back);

// void schedule_front(const bound_data* const data, const int* const permut,const int limit1, int* front);
//
// void schedule_back(const bound_data* const data, const int* const permut,const int limit2, int* back);

void schedule_front(const int* const p_matrix, const int* const min_heads, const int N, const int M, const int* const permut,const int limit1, int* front);

void schedule_back(const int* const p_matrix, const int* const min_tails, const int N, const int M, const int* const permut,const int limit2, int* back);

// void schedule_front(const bound_data* const data, const int* const permut,const int limit1, int* front);
//
// void schedule_back(const bound_data* const data, const int* const permut,const int limit2, int* back);

void sum_unscheduled(const bound_data* const data, const int* const permut, const int limit1, const int limit2, int* remain);

int machine_bound_from_parts(const int* const front, const int* const back, const int* const remain,const int nb_machines);

// void fill_min_heads_tails(const int* const p_times, const int nb_jobs, const int nb_machines, int* min_heads, int* min_tails);

int
add_front_and_bound(const bound_data* const data, const int job, const int * const front, const int * const back, const int * const remain, int* delta_idle);

int
add_back_and_bound(const bound_data* const data, const int job, const int * const front, const int * const back, const int * const remain, int* delta_idle);


void fill_min_heads_tails(bound_data* data, int* min_heads, int* min_tails);

#ifdef __cplusplus
}
#endif

#endif
