#include "c_bound_simple.h"

#include <limits.h>


bound_data* new_bound_data(int _jobs, int _machines)
{
    bound_data *b = malloc(sizeof(bound_data));

    if(b){
        b->p_times = malloc(_jobs*_machines*sizeof(int));
        b->min_heads = malloc(_machines*sizeof(int));
        b->min_tails = malloc(_machines*sizeof(int));
        b->nb_jobs = _jobs;
        b->nb_machines = _machines;
    }

    return b;
}

void free_bound_data(bound_data* b)
{
    if(b){
        free(b->min_tails);
        free(b->min_heads);
        free(b->p_times);
        free(b);
    }
}



inline void
add_forward(const int job, const int * const p_times, const int nb_jobs, const int nb_machines, int * front)
{
    front[0] += p_times[job];
    for (int j = 1; j < nb_machines; j++) {
        front[j] = MAX(front[j - 1], front[j]) + p_times[j * nb_jobs + job];
    }
}


inline void
add_backward(const int job, const int * const p_times, const int nb_jobs, const int nb_machines, int * back)
{
    int j = nb_machines - 1;

    back[j] += p_times[j * nb_jobs + job];
    for (int j = nb_machines - 2; j >= 0; j--) {
        back[j] = MAX(back[j], back[j + 1]) + p_times[j * nb_jobs + job];
    }
}


void
schedule_front(const int* const p_times, const int* const min_heads, const int N, const int M, const int * const permut, const int limit1, int * front)
{
    // int nb_jobs = data->nb_jobs;
    // int nb_machines = data->nb_machines;
    // int *p_times = data->p_times;

    if (limit1 == -1) {
        for (int i = 0; i < M; i++)
            front[i] = min_heads[i];
        return;
    }
    for (int i = 0; i < M; i++){
        front[i]=0;
    }
    for (int i = 0; i < limit1 + 1; i++) {
        add_forward(permut[i], p_times, N, M, front);
    }
}


void
schedule_back(const int* const p_times, const int* const min_tails, const int N, const int M, const int * const permut, const int limit2, int * back)
{
    // int nb_jobs = data->nb_jobs;
    // int nb_machines = data->nb_machines;
    // int *p_times = data->p_times;

    if (limit2 == N) {
        for (int i = 0; i < M; i++)
            back[i] = min_tails[i];
        return;
    }

    for (int i = 0; i < M; i++){
        back[i]=0;
    }
    for (int k = N - 1; k >= limit2; k--) {
        add_backward(permut[k], p_times, N, M, back);
    }
}

void
sum_unscheduled(const bound_data* const data, const int * const permut, const int limit1, const int limit2, int * remain)
{
    int nb_jobs = data->nb_jobs;
    int nb_machines = data->nb_machines;
    int *p_times = data->p_times;

    for (int j = 0; j < nb_machines; j++) {
        remain[j] = 0;
    }
    for (int k = limit1 + 1; k < limit2; k++) {
        int job = permut[k];
        for (int j = 0; j < nb_machines; j++) {
            remain[j] += p_times[j * nb_jobs + job];
        }
    }
}

int
machine_bound_from_parts(const int * const front, const int * const back, const int * const remain,
  const int nb_machines)
{
    int tmp0 = front[0] + remain[0];
    int lb   = tmp0 + back[0]; // LB on machine 0
    int tmp1;

    for (int i = 1; i < nb_machines; i++) {
        tmp1 = MAX(tmp0, front[i] + remain[i]);
        lb   = MAX(lb, tmp1 + back[i]);
        tmp0 = tmp1;
    }

    return lb;
}

int
compute_bound(const bound_data* const data, const int * const permut, const int limit1, const int limit2)
{
    int nb_jobs = data->nb_jobs;
    int nb_machines = data->nb_machines;
    int *p_times = data->p_times;

    int front[nb_machines];
    int back[nb_machines];
    int remain[nb_machines];

    schedule_front(p_times, data->min_heads, nb_jobs, nb_machines, permut, limit1, front);
    schedule_back(p_times, data->min_tails, nb_jobs, nb_machines, permut, limit2, back);
    // schedule_front(data, permut, limit1, front);
    // schedule_back(data, permut, limit2, back);
    sum_unscheduled(data, permut, limit1, limit2, remain);

    int tmp0 = front[0] + remain[0];
    int lb   = tmp0 + back[0]; // LB on machine 0
    int tmp1;

    for (int i = 1; i < nb_machines; i++) {
        tmp1 = MAX(tmp0, front[i] + remain[i]);
        lb   = MAX(lb, tmp1 + back[i]);
        tmp0 = tmp1;
    }

    return lb;
}

// adds job to partial schedule in front and computes lower bound on optimal cost
// NB1: schedule is no longer needed at this point
// NB2: front, remain and back need to be set before calling this
// NB3: also compute total idle time added to partial schedule (can be used a criterion for job ordering)
// nOps : m*(3 add+2 max)  ---> O(m)
int
add_front_and_bound(const bound_data* const data, const int job, const int * const front, const int * const back, const int * const remain, int *delta_idle)
{
    int nb_jobs = data->nb_jobs;
    int nb_machines = data->nb_machines;
    int* p_times = data->p_times;

    int lb   = front[0] + remain[0] + back[0];
    int tmp0 = front[0] + p_times[job];
    int tmp1;

    *delta_idle = 0;
    for (int i = 1; i < nb_machines; i++) {
        *delta_idle += MAX(0, tmp0 - front[i]);

        tmp1 = MAX(tmp0, front[i]);
        lb   = MAX(lb, tmp1 + remain[i] + back[i]);
        tmp0 = tmp1 + p_times[i * nb_jobs + job];
    }

    return lb;
}

// ... same for back
int
add_back_and_bound(const bound_data* const data, const int job, const int * const front, const int * const back, const int * const remain, int *delta_idle)
{
    int nb_jobs = data->nb_jobs;
    int nb_machines = data->nb_machines;
    int* p_times = data->p_times;

    int last_machine = nb_machines - 1;

    int lb   = front[last_machine] + remain[last_machine] + back[last_machine];
    int tmp0 = back[last_machine] + p_times[last_machine*nb_jobs + job];
    int tmp1;

    *delta_idle = 0;
    for (int i = last_machine-1; i >= 0; i--) {
        *delta_idle += MAX(0, tmp0 - back[i]);

        tmp1 = MAX(tmp0, back[i]);
        lb = MAX(lb, tmp1 + remain[i] + front[i]);
        tmp0 = tmp1 + p_times[i*nb_jobs + job];
    }

    return lb;
}

void
fill_min_heads_tails(bound_data* data, int * min_heads, int * min_tails)
{
    int nb_machines = data->nb_machines;
    int nb_jobs = data->nb_jobs;
    int *p_times = data->p_times;

    int tmp[nb_machines];

    // 1/ min start times on each machine
    for (int k = 0; k < nb_machines; k++) min_heads[k] = INT_MAX;
    min_heads[0] = 0; // per definition =0 on first machine

    for (int i = 0; i < nb_jobs; i++) {
        for (int k = 0; k < nb_machines; k++)
            tmp[k] = 0;

        tmp[0] += p_times[i];
        for (int k = 1; k < nb_machines; k++) {
            tmp[k] = tmp[k - 1] + p_times[k * nb_jobs + i];
        }

        for (int k = 1; k < nb_machines; k++) {
            min_heads[k] = MIN(min_heads[k], tmp[k - 1]);
        }
    }

    // 2/ min run-out times on each machine
    for (int k = 0; k < nb_machines; k++) min_tails[k] = INT_MAX;
    // per definition =0 on last machine
    min_tails[nb_machines - 1] = 0;

    for (int i = 0; i < nb_jobs; i++) {
        for (int k = 0; k < nb_machines; k++)
            tmp[k] = 0;

        tmp[nb_machines - 1] += p_times[(nb_machines - 1) * nb_jobs + i];

        for (int k = nb_machines - 2; k >= 0; k--) {
            tmp[k] = tmp[k + 1] + p_times[k * nb_jobs + i];
        }
        for (int k = nb_machines - 2; k >= 0; k--) {
            min_tails[k] = MIN(min_tails[k], tmp[k + 1]);
        }
    }
} /* fill_min_heads_tails */


// #ifdef __cplusplus
// }
// #endif
