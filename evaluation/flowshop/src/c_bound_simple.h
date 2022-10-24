#ifdef __cplusplus
extern "C" {
#endif

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

inline void add_forward(const int job, const int* const p_times, const int nb_jobs, const int nb_machines, int *front)
{
    front[0] += p_times[job];
    for(int j=1;j<nb_machines; j++){
        front[j] = MAX(front[j - 1], front[j]) + p_times[j*nb_jobs + job];
    }
}

void schedule_front(const int* const permut,const int limit1, const int* const p_times, const int nb_jobs, const int nb_machines, int* front)
{
    for (int i = 0; i < limit1 + 1; i++) {
        add_forward(permut[i],p_times,nb_jobs,nb_machines,front);
    }
}


inline void add_backward(const int job, const int* const p_times, const int nb_jobs, const int nb_machines, int *back)
{
    int j=nb_machines-1;

    back[j] += p_times[j*nb_jobs + job];
    for (int j = nb_machines - 2; j >= 0; j--) {
        back[j] = MAX(back[j], back[j + 1]) + p_times[j*nb_jobs + job];
    }
}

void schedule_back(const int* const permut,const int limit2, const int* const p_times, const int nb_jobs, const int nb_machines, int* back)
{
    for (int k = nb_jobs - 1; k >= limit2; k--) {
        add_backward(permut[k],p_times,nb_jobs,nb_machines,back);
    }
}

void sum_unscheduled(const int* const permut, const int limit1, const int limit2, const int* const p_times, const int nb_jobs, const int nb_machines, int* remain)
{
    for (int j = 0; j < nb_machines; j++) {
        remain[j]=0;
    }
    for (int k = limit1 + 1; k < limit2; k++) {
        int job = permut[k];
        for (int j = 0; j < nb_machines; j++) {
            remain[j] += p_times[j*nb_jobs+job];
        }
    }
}

int machine_bound_from_parts(const int* const front, const int* const back, const int* const remain,const int nb_machines)
{
    int tmp0 = front[0] + remain[0];
    int lb = tmp0 + back[0]; //LB on machine 0
    int tmp1;

    for(int i=1;i<nb_machines;i++)
    {
        tmp1 = MAX(tmp0,front[i] + remain[i]);
        lb = MAX(lb,tmp1+back[i]);
        tmp0 = tmp1;
    }

    return lb;
}


int compute_bound(const int* const permut,const int limit1, const int limit2, const int* const p_times, const int nb_jobs, const int nb_machines)
{
    int front[nb_machines];
    int back[nb_machines];
    int remain[nb_machines];

    schedule_front(permut,limit1,p_times,nb_jobs,nb_machines,front);
    schedule_back(permut,limit2,p_times,nb_jobs,nb_machines,back);
    sum_unscheduled(permut,limit1,limit2,p_times,nb_jobs,nb_machines,remain);

    int tmp0 = front[0] + remain[0];
    int lb = tmp0 + back[0]; //LB on machine 0
    int tmp1;

    for(int i=1;i<nb_machines;i++)
    {
        tmp1 = MAX(tmp0,front[i] + remain[i]);
        lb = MAX(lb,tmp1+back[i]);
        tmp0 = tmp1;
    }

    return lb;
}

int add_front_and_bound(const int job, const int* const front,const int* const back, const int* const remain, const int* const p_times, const int nb_jobs, const int nb_machines)
{
    int lb = front[0]+remain[0]+back[0];
    int tmp0 = front[0] + p_times[job];
    int tmp1;

    for(int i=1;i<nb_machines;i++){
        tmp1 = MAX(tmp0, front[i]);
        lb = MAX(lb,tmp1 + remain[i] + back[i]);
        tmp0 = tmp1 + p_times[i*nb_jobs + job];
    }

    return lb;
}



#ifdef __cplusplus
}
#endif
