#include <stdlib.h>
#include <stdio.h>

#include "c_taillard.h"
#include "c_bound_simple.h"
#include "c_bound_johnson.h"

int main(int argc,char **argv)
{
    int inst_id = atoi(argv[1]);

    //------------------simple bound------------------
    int N = taillard_get_nb_jobs(inst_id);
    int M = taillard_get_nb_machines(inst_id);

    //allocate memory
    bound_data* lb1_data = new_bound_data(N,M);

    //fill data struct for LB1
    taillard_get_processing_times(lb1_data->p_times,inst_id);
    fill_min_heads_tails(lb1_data);

    //---------------a permutation------------------------
    int permutation[N];
    for(int i=0;i<N;i++){ //assume 0-based permutation
        permutation[i]=i;
    }

    //---------------------test---------------------
    printf("===============SIMPLE BOUND====================\n");

    //evaluate solution
    int cmax = eval_solution(lb1_data,permutation);
    printf("cmax:\t%d\n",cmax);

    //eval bounds one by one
    for(int lim1=0;lim1<N;lim1++){
        int lb = lb1_bound(lb1_data,permutation,lim1,N);
        printf("lb[%d,N]:\t%d\n",lim1,lb);
    }

    //evaluate children
    int lim1 = -1; //limits of the parent problem (root)
    int lim2 = N;

    int lb_begin[N];
    int lb_end[N];

    enum direction { BEGIN=-1, BEGINEND=0, END=1};

    lb1_children_bounds(lb1_data, permutation, lim1, lim2, lb_begin, lb_end, NULL, NULL, BEGINEND);

    for(int i=0;i<N;i++){
        printf("%2d/...: %d\t\t.../%2d: %d\n",i,lb_begin[i],i,lb_end[i]);
    }


    printf("===============JOHNSON=========================\n");
    // enum lb2_variant lb2_type = FULL;

    //allocate
    johnson_bd_data* lb2_data = new_johnson_bd_data(lb1_data,LB2_FULL);

    fill_machine_pairs(lb2_data,LB2_FULL);
    fill_lags(lb1_data,lb2_data);
    fill_johnson_schedules(lb1_data,lb2_data);

    int best_cmax = 2147483647;

    //eval bounds one by one
    for(int lim1=0;lim1<N;lim1++){
        int lb = lb2_bound(lb1_data,lb2_data,permutation,lim1,N,best_cmax);
        printf("lb[%d,N]:\t%d\n",lim1,lb);
    }

    lim1 = -1; //limits of the parent problem (root)
    lim2 = N;
    lb2_children_bounds(lb1_data, lb2_data, permutation, lim1, lim2, lb_begin, lb_end, best_cmax, BEGINEND);

    for(int i=0;i<N;i++){
        printf("%2d/...: %d\t\t.../%2d: %d\n",i,lb_begin[i],i,lb_end[i]);
    }


    //free memory
    free_bound_data(lb1_data);
    free_johnson_bd_data(lb2_data);

}
