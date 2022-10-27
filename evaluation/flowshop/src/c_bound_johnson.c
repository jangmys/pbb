#include <stdlib.h>
#include <stdio.h>

#include "c_bound_simple.h"
#include "c_bound_johnson.h"

#ifdef __cplusplus
extern "C" {
#endif

johnson_bd_data* new_johnson_bd_data(bound_data *data)
{
    johnson_bd_data *b = malloc(sizeof(johnson_bd_data));

    b->nb_jobs = data->nb_jobs;
    b->nb_machines = data->nb_machines;
    b->nb_machine_pairs = ((b->nb_machines-1)*b->nb_machines)/2;

    b->lags = malloc(b->nb_machine_pairs*b->nb_jobs*sizeof(int));
    b->johnson_schedules = malloc(b->nb_machine_pairs*b->nb_jobs*sizeof(int));
    b->machine_pairs[0] = malloc(b->nb_machine_pairs*sizeof(int));
    b->machine_pairs[1] = malloc(b->nb_machine_pairs*sizeof(int));
    b->machine_pair_order = malloc(b->nb_machine_pairs*sizeof(int));
    return b;
}

void free_johnson_bd_data(johnson_bd_data* b)
{
    if(b){
        free(b->lags);
        free(b->johnson_schedules);
        free(b->machine_pairs[0]);
        free(b->machine_pairs[1]);
        free(b->machine_pair_order);
        free(b);
    }
}

void fill_machine_pairs(johnson_bd_data* b)
{
    if(!b)
    {
        printf("allocate johnson_bd_data first\n");
        exit(-1);
    }

    unsigned c=0;
    for(int i=0; i<b->nb_machines-1; i++){
        for(int j=i+1; j<b->nb_machines; j++){
            b->machine_pairs[0][c]=i;
            b->machine_pairs[1][c]=j;
            b->machine_pair_order[c]=c;
            c++;
        }
    }
}



// term q_iuv in [Lageweg'78]
void
fill_lags(const int * const p_times, const int nb_jobs, const int nb_machines, int * lags)
{
    int count = 0;

    for (int m1 = 0; m1 < nb_machines - 1; m1++) {
        for (int m2 = m1 + 1; m2 < nb_machines; m2++) {
            for (int j = 0; j < nb_jobs; j++) {
                lags[count * nb_jobs + j] = 0;
                for (int k = m1 + 1; k < m2; k++) {
                    lags[count * nb_jobs + j] += p_times[k * nb_jobs + j];
                }
            }
            count++;
        }
    }
}

typedef struct johnson_job{
    int job; //job-id
    int partition; //in partition 0 or 1
    int ptm1; //processing time on m1
    int ptm2; //processing time on m2
} johnson_job;

//(after partitioning) sorting jobs in ascending order with this comparator yield an optimal schedule for the associated 2-machine FSP [Johnson, S. M. (1954). Optimal two-and three-stage production schedules with setup times included.closed access Naval research logistics quarterly, 1(1), 61–68.]
int johnson_comp(const void * elem1, const void * elem2)
{
    johnson_job j1 = *((johnson_job*)elem1);
    johnson_job j2 = *((johnson_job*)elem2);

    //partition 0 before 1
    if(j1.partition == 0 && j2.partition == 1)return -1;
    if(j1.partition == 1 && j2.partition == 0)return 1;

    //in partition 0 increasing value of ptm1
    if(j1.partition == 0){
        if(j2.partition == 1)return -1;
        return j1.ptm1 - j2.ptm1;
    }
    //in partition 1 decreasing value of ptm1
    if(j1.partition == 1){
        if(j2.partition == 0)return 1;
        return j2.ptm2 - j1.ptm2;
    }
}

//for each machine-pair (m1,m2), solve 2-machine FSP with processing times
//  p_1i = PTM[m1][i] + lags[s][i]
//  p_2i = PTM[m2][i] + lags[s][i]
//using Johnson's algorithm [Johnson, S. M. (1954). Optimal two-and three-stage production schedules with setup times included.closed access Naval research logistics quarterly, 1(1), 61–68.]
void fill_johnson_schedules(const int* const p_times, const int* const lags, const int nb_jobs, const int nb_machines, int* johnson_schedules)
{
    johnson_job tmp[nb_jobs];

    int s=0;
    for(int m1=0; m1<nb_machines-1;m1++){
        for(int m2=m1+1;m2<nb_machines;m2++){
            //partition jobs into 2 sets {j|p_1j < p_2j} and {j|p_1j >= p_2j}
            for(int i=0; i<nb_jobs; i++){
                tmp[i].job = i;
                tmp[i].ptm1 = p_times[m1*nb_jobs + i] + lags[s*nb_jobs + i];
                tmp[i].ptm2 = p_times[m2*nb_jobs + i] + lags[s*nb_jobs + i];

                if(tmp[i].ptm1<tmp[i].ptm2){
                    tmp[i].partition=0;
                }else{
                    tmp[i].partition=1;
                }
            }
            //sort according to johnson's criterion
            qsort (tmp, sizeof(tmp)/sizeof(*tmp), sizeof(*tmp), johnson_comp);

            for(int i=0; i<nb_jobs; i++){
                johnson_schedules[s*nb_jobs + i] = tmp[i].job;
            }
            s++;
        }
    }
}


inline int compute_cmax_johnson(const bound_data* const bd, const johnson_bd_data* const jhnsn, const int* const flag, int *tmp0, int *tmp1, int ma0, int ma1, int ind)
{
    int nb_jobs = bd->nb_jobs;

    for (int j = 0; j < nb_jobs; j++) {
        int job = jhnsn->johnson_schedules[ind*nb_jobs + j];
        // j-loop is on unscheduled jobs... (==0 if jobCour is unscheduled)
        if (flag[job] == 0) {
            int ptm0 = bd->p_times[ma0*nb_jobs + job];
            int ptm1 = bd->p_times[ma1*nb_jobs + job];
            int lag = jhnsn->lags[ind*nb_jobs + job];
            // add job on ma0 and ma1
            *tmp0 += ptm0;
            *tmp1 = MAX(*tmp1,*tmp0 + lag);
            *tmp1 += ptm1;
        }
    }

    return *tmp1;
}


int lb_makespan(const bound_data* const bd, const johnson_bd_data* const jhnsn, const int* const flag, const int* const front, const int* const back, const int minCmax){
    int lb=0;

    // for all machine-pairs : O(m^2) m*(m-1)/2
    for (int l = 0; l < jhnsn->nb_machine_pairs; l++) {
        int i = jhnsn->machine_pair_order[l];

        int ma0 = jhnsn->machine_pairs[0][i];
        int ma1 = jhnsn->machine_pairs[1][i];

        int tmp0 = front[ma0];
        int tmp1 = front[ma1];

        compute_cmax_johnson(bd,jhnsn,flag,&tmp0,&tmp1,ma0,ma1,i);

        tmp1 = MAX(tmp1 + back[ma1], tmp0 + back[ma0]);

        lb=MAX(lb,tmp1);

        if(lb>minCmax){
            break;
        }
    }

    return lb;
}


#ifdef __cplusplus
}
#endif
