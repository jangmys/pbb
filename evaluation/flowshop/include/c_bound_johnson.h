#ifdef __cplusplus
extern "C" {
#endif

#include "c_bound_simple.h"

//regroup (constant) bound data
typedef struct johnson_bd_data
{
    int *johnson_schedules;
    int *lags;
    int *machine_pairs[2];
    int *machine_pair_order;

    int nb_machine_pairs;
    int nb_jobs;
    int nb_machines;
} johnson_bd_data;

johnson_bd_data* new_johnson_bd_data(bound_data* lb1);
void free_johnson_bd_data(johnson_bd_data* b);

// void allocate(const int N, const int M);

void fill_machine_pairs(johnson_bd_data* b);

void fill_lags(const int* const p_times, const int nb_jobs, const int nb_machines, int* lags);

void fill_johnson_schedules(const int* const p_times, const int* const lags, const int nb_jobs, const int nb_machines, int* johnson_schedules);

int compute_cmax_johnson(const bound_data* const bd, const johnson_bd_data* const jhnsn, const int* const flag, int* tmp0, int* tmp1, int ma0, int ma1, int ind);

int lb_makespan(const bound_data* const bd, const johnson_bd_data* const jhnsn, const int* const flag, const int* const front, const int* const back, const int minCmax);


#ifdef __cplusplus
}
#endif
