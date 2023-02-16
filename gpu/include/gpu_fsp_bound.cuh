#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

//for 40-machine instances...
#define MAXMACH 40
#define MAXSOMME 780 //M*(M-1)/2

#define TILE_SZ 32
#define MAXJOBS 800

__device__ unsigned int _trigger;

__device__ int root_d[MAXJOBS];
__device__ int root_dir_d;

// constant GPU data
__device__ __constant__ int _nbMachines;
__device__ __constant__ int _sum;
__device__ __constant__ int _nbJobPairs;

__device__ __constant__ int _sumPT[MAXMACH];            //20
__device__ __constant__ int _minTempsArr[MAXMACH];            //20
__device__ __constant__ int _minTempsDep[MAXMACH];            //20


#if MAXJOBS == 20
__device__ __constant__ int _tempsJob[MAXJOBS * MAXMACH];     //400
__device__ __constant__ int _tabJohnson[MAXJOBS * MAXSOMME];  //3800
__device__ __constant__ int _tempsLag[MAXJOBS * MAXSOMME];    //3800
__device__ __constant__ int _machine[2 * MAXSOMME];           //380
//4*(7600+760+400+40)=35200 B constant memory
#elif MAXJOBS == 50
__device__ __constant__ int _tempsJob[MAXJOBS * MAXMACH];     //1000
__device__ __constant__ int _tabJohnson[MAXJOBS * MAXSOMME];  //9500
__device__ __constant__ int _machine[2 * MAXSOMME];           //380
__device__ int _tempsLag[MAXJOBS * MAXSOMME];
//4*(9500+1000+380+40)=43680 B constant memory
//38000 B global mem
#elif MAXJOBS == 100
__device__ __constant__ int _tempsJob[MAXJOBS * MAXMACH];     //2000
__device__ __constant__ int _machine[2 * MAXSOMME];           //380
__device__ int _tabJohnson[MAXJOBS * MAXSOMME];  //19000
__device__ int _tempsLag[MAXJOBS * MAXSOMME];   //19000
//__device__ __constant__ int _tabJohnson[MAXJOBS * MAXSOMME];  //9500
//4*38k=152kB gmem
#elif MAXJOBS == 200
__device__ __constant__ int _tempsJob[MAXJOBS * MAXMACH];     //4000
__device__ __constant__ int _machine[2 * MAXSOMME];           //380
__device__ int _tabJohnson[MAXJOBS * MAXSOMME];  //9500
__device__ int _tempsLag[MAXJOBS * MAXSOMME];   //38000
//__device__ __constant__ int _tempsLag[MAXJOBS * MAXSOMME];    //9500
//4*76k=304kB gmem
#elif MAXJOBS >= 300
__device__ int _tempsJob[MAXJOBS * MAXMACH];     //10000
__device__ __constant__ int _machine[2 * MAXSOMME];           //380
__device__ int _tabJohnson[MAXJOBS * MAXSOMME];  //95000
__device__ int _tempsLag[MAXJOBS * MAXSOMME];   //95000


//4*(10420)=41680 B const
//int *tempsLag_d;
//texture<int>tempsLag_tex;
#endif

__device__ int _jobPairs[MAXJOBS*(MAXJOBS-1)];
__device__ int freqTable_d[MAXJOBS * MAXJOBS];

// bounding
int nbMachines_h;
int nbJob_h;
int somme_h;
int nbJobPairs_h;

int *tempsJob_h;
int *tabJohnson_h;
int *tempsLag_h;
int *minTempsArr_h;
int *minTempsDep_h;
int *sumPT_h;

int *machine_h;
int *jobPairs_h;

void
allocate_host_bound_tmp()
{
    tempsJob_h    = (int *) malloc(nbMachines_h * nbJob_h * sizeof(int));
    tabJohnson_h  = (int *) malloc(nbJob_h * somme_h * sizeof(int));
    tempsLag_h    = (int *) malloc(nbJob_h * somme_h * sizeof(int));
    minTempsDep_h = (int *) malloc(nbMachines_h * sizeof(int));
    minTempsArr_h = (int *) malloc(nbMachines_h * sizeof(int));

    machine_h  = (int *) malloc(2 * somme_h * sizeof(int));
    jobPairs_h = (int *) malloc(2 * nbJobPairs_h * sizeof(int));

    sumPT_h = (int *) malloc(nbMachines_h * sizeof(int));
}

void free_host_bound_tmp(){
  free(tempsJob_h);
  free(tabJohnson_h);
  free(tempsLag_h);
  free(minTempsDep_h);
  free(minTempsArr_h);
  free(machine_h);
  free(jobPairs_h);
  free(sumPT_h);
}

//==============================
//HOST FUNCTIONS
//PREPARING BOUNDING DATA ON CPU
//==============================
/**
 * minTempsDep/minTempsArr
 * earlist possible starting time of a job on machines and
 * shortest possible completion time of a job after release from machines
 * size : nbMachines
 * requires : PTM
 */
void fillMinTempsArrDep(){
    for (int k = 0; k < nbMachines_h; k++) {
        minTempsDep_h[k]=9999999;
    }
    minTempsDep_h[nbMachines_h-1]=0;
    int *tmp=new int[nbMachines_h];

    for (int i = 0; i<nbJob_h; i++){
       for (int k = nbMachines_h-1; k>=0; k--) {
           tmp[k]=0;
       }
       tmp[nbMachines_h-1]+=tempsJob_h[(nbMachines_h-1)*nbJob_h + i];//ptm[(nbMachines-1) * nbJob + job];
       for (int k = nbMachines_h - 2; k >= 0; k--){
           tmp[k]=tmp[k+1]+tempsJob_h[k*nbJob_h + i];
       }
       for (int k = nbMachines_h-2; k>=0; k--) {
           minTempsDep_h[k]=(tmp[k+1]<minTempsDep_h[k])?tmp[k+1]:minTempsDep_h[k];
       }
    }

    for (int k = 0; k < nbMachines_h; k++) {
       minTempsArr_h[k]=9999999;
    }
    minTempsArr_h[0]=0;

    for (int i = 0; i < nbJob_h; i++) {
       for (int k = 0; k < nbMachines_h; k++) {
           tmp[k]=0;
       }
       tmp[0]+=tempsJob_h[i];
       for (int k = 1; k < nbMachines_h; k++) {
           tmp[k]=tmp[k-1]+tempsJob_h[k*nbJob_h+i];
       }
       for (int k = 1; k < nbMachines_h; k++) {
           minTempsArr_h[k]=(tmp[k-1]<minTempsArr_h[k])?tmp[k-1]:minTempsArr_h[k];
       }
    }

    delete[]tmp;
}

void fillSumPT()
{
    for (int k = 0; k < nbMachines_h; k++) {
        sumPT_h[k]=0;
        for (int i = 0; i < nbJob_h; i++) {
            sumPT_h[k] += tempsJob_h[k*nbJob_h + i];
        }
    }
}

/**
 * array machine_h contains all possible machine-pairs...
 * indices of upper triangular matrix
 * size : 2*n(n-1)/2 integers
 */
void fillMachine()
{
	int cmpt = 0;
	for (int i = 0; i < (nbMachines_h - 1); i++){
		for (int j = i + 1; j < nbMachines_h; j++) {
			machine_h[cmpt] = i;
			machine_h[somme_h + cmpt] = j;
			cmpt++;
		}
    }

    cmpt=0;
	for (int i = 0; i < (nbJob_h - 1); i++){
		for (int j = i + 1; j < nbJob_h; j++) {
			jobPairs_h[cmpt] = i;
			jobPairs_h[nbJobPairs_h + cmpt] = j;
			cmpt++;
		}
    }

}
/**
 * for each couple of machines (m1,m2) and all jobs j
 * compute (min) time required for job to be processed on machines m1<...<m2
 * requires : PTM
 * required in : FSP lowre bound (Johnson)
 * size : n*(n*(n-1)/2) integers
 */
void fillLag(){
  int m1, m2;

  for (int i = 0; i < somme_h; i++) {
    m1 = machine_h[i];
    m2 = machine_h[somme_h + i];

    for (int j = 0; j < nbJob_h; j++) {
      tempsLag_h[i*nbJob_h + j] = 0;

      for (int k = m1 + 1; k < m2; k++)
        tempsLag_h[i*nbJob_h + j] += tempsJob_h[k*nbJob_h + j];
    }
  }
}

////////////////////////////////////........johnson
int *pluspetit[2];

int estInf(int i, int j) {
  if (pluspetit[0][i] == pluspetit[0][j]) {
    if (pluspetit[0][i] == 1)
      return pluspetit[1][i] < pluspetit[1][j];

    return pluspetit[1][i] > pluspetit[1][j];
  }
  return pluspetit[0][i] < pluspetit[0][j];
}
int estSup(int i, int j) {
  if (pluspetit[0][i] == pluspetit[0][j]) {
    if (pluspetit[0][i] == 1)
      return pluspetit[1][i] > pluspetit[1][j];

    return pluspetit[1][i] < pluspetit[1][j];
  }
  return pluspetit[0][i] > pluspetit[0][j];
}
int partionner(int *ordo, int deb, int fin) {
  int d = deb - 1;
  int f = fin + 1;
  int mem, pivot = ordo[deb];

  do {
    do
      f--;
    while (estSup(ordo[f], pivot));

    do
      d++;
    while (estInf(ordo[d], pivot));

    if (d < f) {
      mem = ordo[d];
      ordo[d] = ordo[f];
      ordo[f] = mem;
    }
  } while (d < f);
  return f;
}
void quicksort(int *ordo, int deb, int fin) {
  int k;

  if ((fin - deb) > 0) {
    k = partionner(ordo, deb, fin);
    quicksort(ordo, deb, k);
    quicksort(ordo, k + 1, fin);
  }
}
void Johnson(int *ordo, int m1, int m2, int s) {
  pluspetit[0] = (int *)malloc((nbJob_h) * sizeof(int));
  pluspetit[1] = (int *)malloc((nbJob_h) * sizeof(int));

  for (int i = 0; i < nbJob_h; i++) {
    ordo[i] = i;

    if (tempsJob_h[m1*nbJob_h + i] < tempsJob_h[m2*nbJob_h + i]) {
      pluspetit[0][i] = 1;
      pluspetit[1][i] = tempsJob_h[m1*nbJob_h + i] + tempsLag_h[s*nbJob_h + i];
    } else {
      pluspetit[0][i] = 2;
      pluspetit[1][i] = tempsJob_h[m2*nbJob_h + i] + tempsLag_h[s*nbJob_h + i];
    }
  }
  quicksort(ordo, 0, (nbJob_h - 1));

  free(pluspetit[0]);
  free(pluspetit[1]);
}
//SUM*JOBS
void fillTabJohnson()
{
	int cmpt = 0;

	for (int i = 0; i < (nbMachines_h - 1); i++)
		for (int j = i + 1; j < nbMachines_h; j++) {
			Johnson(tabJohnson_h+cmpt*nbJob_h, i, j, cmpt);
			cmpt++;
		}
}


/*********************************
 ****** evaluate bounds **********
 *********************************/
//§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
#include "gpu_lb_johnson.cuh"


//§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
//thread evaluates
//swap(permutation,toSwap1,toSwap2)
template <typename T>
inline __device__ int
thread_evalSol_d(const T * permutation, const T toSwap1, const T toSwap2)
{
    int temps[MAXMACH] = { 0 };
    int job;

    for (int mm = 0; mm < _nbMachines; mm++)
        temps[mm] = 0;

    for (int j = 0; j < size_d; j++) {
        if (j == toSwap1)
            job = permutation[toSwap2];
        else if (j == toSwap2)
            job = permutation[toSwap1];
        else
            job = permutation[j];

        temps[0] = temps[0] + _tempsJob[job];

        for (int m = 1; m < _nbMachines; m++)
            temps[m] = max(temps[m], temps[m - 1]) + _tempsJob[m * size_d + job];
    }

    return temps[_nbMachines - 1];
}

//insert {toSwap1 < toSwap2}
template <typename T>
inline __device__ int
thread_evalSol_insert_d(const T *  permutation, const T toSwap1, const T toSwap2) {
  int temps[MAXMACH]={0};
  int job;

  for (int mm = 0; mm < _nbMachines; mm++)
    temps[mm] = 0;

  for (int j = 0; j < size_d; j++) {
    if (j == toSwap1)
      job = permutation[toSwap2];
    else if (j > toSwap1 && j<= toSwap2)
      job = permutation[j-1];
    else
      job = permutation[j];

    temps[0] = temps[0] + _tempsJob[job];

    for (int m = 1; m < _nbMachines; m++)
      temps[m] = max(temps[m], temps[m - 1]) + _tempsJob[m * size_d + job];
  }

  return temps[_nbMachines - 1];
}

//§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§

template <typename T>
inline __device__ int
computeCostFast(const T* schedules_sh, const T toSwap1, const T toSwap2, const T limit1, const T limit2, const unsigned char * tempsJob, const int mat_id, const int * _tabJohnson, const int* front, const int* back, const int best) {
  int tempsMachines[MAXMACH]={0};
  int tempsMachinesFin[MAXMACH]={0};

  int job[MAXJOBS]={0};
  // int jobFin[MAXJOBS]={0};

  int borneInf=0;

  if (limit2 - limit1 == 1) {
    borneInf=thread_evalSol_d(schedules_sh+mat_id*size_d, toSwap1, toSwap2);
  } else {
    set_tempsMachines_d(front, tempsMachines, schedules_sh, toSwap1, toSwap2, limit1, _tempsJob, mat_id); //front
    set_job_d(job, schedules_sh, toSwap1, toSwap2, limit1,limit2, mat_id);
    set_tempsMachinesFin_d(back,schedules_sh+mat_id*size_d, tempsMachinesFin, toSwap1, toSwap2, limit2, _tempsJob);

    borneInf=calculBorne_d(job, tempsMachinesFin, tempsMachines, limit1 + 1, size_d - limit2, _tempsJob, _tabJohnson, best);
  }
  return borneInf;
}

//JOHNSON BOUND
template <typename T>
inline __device__ int
computeCost(const T* schedules_sh, const T toSwap1, const T toSwap2, const T limit1, const T limit2, const int * tempsJob, const int mat_id, const int * _tabJohnson, const int best) {
  int tempsMachines[MAXMACH]={0};
  int tempsMachinesFin[MAXMACH]={0};

  int job[MAXJOBS]={0};
  int jobFin[MAXJOBS]={0};

  int nbAffectFin = size_d - limit2;
  int nbAffectDebut = limit1 + 1;
  int borneInf=0;

  if (limit2 - limit1 == 1) {
    borneInf=thread_evalSol_d(schedules_sh+mat_id*size_d, toSwap1, toSwap2);
  } else {
    set_tempsMachines_retardDebut_d(tempsMachines, schedules_sh, toSwap1,
                                    toSwap2, limit1, _tempsJob, mat_id);
    set_job_jobFin_d(job, jobFin, schedules_sh, toSwap1, toSwap2, limit1,
                     limit2, mat_id);
    set_tempsMachinesFin_tempsJobFin_d(jobFin, tempsMachinesFin, nbAffectFin,_tempsJob);
    borneInf=calculBorne_d(job, tempsMachinesFin, tempsMachines, nbAffectDebut,
                      nbAffectFin, _tempsJob, _tabJohnson, best);
  }
  return borneInf;
}

template <typename T>
__global__ void
__launch_bounds__(128, 8) bound(const T * schedules_d, const T * limit1s_d, const T * limit2s_d, const T * line_d,
  int * costsBE_d, int * sums_d, const T * state_d, const int * toSwap_d,
  const int * ivmId_d, unsigned int * bdleaves_d, unsigned int * ctrl_d, int * flagLeaf,
  const int best){
    /**** thread indexing ****/
    register int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    register int BE    = tid & 1;
    register int ivmnb = ivmId_d[(tid >> 1)]; // the ivm tid is working on

    /***** shared memory declarations *****/
    extern __shared__ unsigned char sharedArray[];
    unsigned char * tempsJob_sh = (unsigned char *) sharedArray;
    char * permut_sh = (char *) &tempsJob_sh[_nbMachines * size_d];

    if (threadIdx.x < size_d) {
        for (int j = 0; j < _nbMachines; j++)
            tempsJob_sh[j * size_d + threadIdx.x] =
              (unsigned char) _tempsJob[j * size_d + threadIdx.x];
    }
    if (tid < 2 * ctrl_d[toDo]) {
        //  if (tid < 2 * ctrl_d[0]) {
        if (tid % 2 == 0) {
            for (int i = 0; i < size_d; i++)
                permut_sh[index2D(i, threadIdx.x >> 1)] =
                  schedules_d[index2D(i, ivmnb)];
        }
    }

    __syncthreads();
    /*******************************************/
    if (tid < 2 * ctrl_d[toDo]) {
        //  if (tid < 2 * ctrl_d[0]) {
        char limit1 = limit1s_d[ivmnb] + 1 - BE;
        char limit2 = limit2s_d[ivmnb] - BE;

        char Swap1 = toSwap_d[(tid >> 1)];
        char Swap2 = (1 - BE) * limit1 + BE * limit2;

        char jobnb = permut_sh[index2D(Swap1, threadIdx.x >> 1)];

        int where = ivmnb * 2 * size_d + BE * size_d + (int) jobnb;

        if (line_d[ivmnb] < (size_d - 1)) { // boundNodes
            costsBE_d[where] = computeCost(permut_sh, Swap1, Swap2, limit1, limit2, _tempsJob, threadIdx.x>>1,_tabJohnson,best);// + BE * limit2;
            // costsBE_d[where] = computeCost(permut_sh, Swap1, Swap2, limit1, limit2, tempsJob_sh, threadIdx.x>>1,_tabJohnson,best);// + BE * limit2;
            atomicAdd(&sums_d[2 * ivmnb + BE], costsBE_d[where]);
        } else if (BE == 0) { // boundLeaves
            if (state_d[ivmnb] == 1)
                bdleaves_d[ivmnb]++;

            flagLeaf[ivmnb] = 1;
            atomicInc(&ctrl_d[foundLeaf], UINT_MAX);
        }
    }
}

//§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
//warp parallel partial makespan evaluation : front
//input : permutation, limit1, processing time matrix
//output : completion times of prefix schedule on all machines / remaining
template <unsigned tile_size>
inline __device__ void
tile_scheduleFront(thread_block_tile<tile_size> g, const int* permutation, const int limit1, const int* ptm, int *front, int *remain) {
    int lane = g.thread_rank();

	if(limit1 == -1){
        for (int i = lane; i < _nbMachines; i+=tile_size) {
            front[i]=_minTempsArr[i];
        }
        return;
	}

    for (int i = lane; i < _nbMachines; i+=tile_size)
        front[i] = 0;

    int slice_width = min(tile_size,limit1+1);
    int n_slices = 1 + limit1/tile_size;

    for(int k=0;k<n_slices;k++){
        int to = min(slice_width,limit1+1-k*slice_width);

        int pt;
        int job;
        int tmp0=0;
        int tmp1=0;
        if(lane<_nbMachines){
            tmp0=front[lane];
            tmp1=tmp0;
        }
        for(int i=0;i<to;i++){ //for jobs scheduled in front
            tmp1=g.shfl_up(tmp0,1); //no wrap-around (lowest lane not modified)
            if(lane<min(i+1,_nbMachines)){
                job=permutation[k*slice_width+i-lane];
                pt = ptm[lane*size_d+job];
                remain[lane] -= pt;
                tmp0=max(tmp0,tmp1)+pt;
            }
        }
        if(lane==0)
            front[0]=tmp0;

        if(lane<to){
            job=permutation[to+k*slice_width-lane-1];
        }

        for(int i=1;i<_nbMachines;i++){
            tmp1=g.shfl_down(tmp0,1);
            //top-thread needs to read from memory
            if(k>0 && lane == to-1 && lane+1 < _nbMachines)
                tmp1 = front[lane+i];

            if(lane<min(_nbMachines-i,to)){ // mini(_nbMachines-i,mNM)
                pt = ptm[(lane+i)*size_d+job];
                remain[lane+i] -= pt;
                tmp0=max(tmp0,tmp1)+pt;
            }
            if(lane==0)front[i]=tmp0;
        }
    }
}

template <unsigned tile_size>
inline __device__ void
tile_scheduleBack(thread_block_tile<tile_size> g, const int* permutation, const int limit2, const int* ptm, int *back, int *remain)
{
    int lane = g.thread_rank();

    if(limit2==size_d){
        for(int i=lane; i<_nbMachines; i+=tile_size)
            back[i]=_minTempsDep[i];
        return;
    }
    for (int i = lane; i < _nbMachines; i+=tile_size)
        back[i] = 0;

    int slice_width = min(tile_size,size_d-limit2);
    int n_slices = 1 + (size_d-limit2-1)/tile_size;

    for(int k=0;k<n_slices;k++){
        int to = min(slice_width,(size_d-limit2)-k*slice_width);

        int pt;
        int job;
        int tmp0=0;
        int tmp1=0;
        if(lane<_nbMachines){
            tmp0=back[_nbMachines-1-lane];
            tmp1=tmp0;
        }

        int ma=_nbMachines-1-lane;

        for(int i=0;i<to;i++){ //for jobs scheduled in back
            tmp1=g.shfl_up(tmp0,1);
            if( lane < min(i+1,_nbMachines) ){
                job=permutation[(size_d-1)+lane-i-k*slice_width];
                pt=ptm[ma*size_d+job];
                remain[ma] -= pt;
                tmp0=max(tmp0,tmp1)+pt;
            }
        }

        if(lane==0)
            back[_nbMachines-1]=tmp0;

        if(lane<to){
            job=permutation[size_d-(to-lane)-k*slice_width];
        }

        for(int i=1;i<_nbMachines;i++){
            ma--;
            tmp1=g.shfl_down(tmp0,1);
            //top-thread needs to read from memory
            if(k>0 && lane == to-1 && lane+1 < _nbMachines)
                tmp1 = back[_nbMachines-1-(lane+i)];

            if( lane<min(_nbMachines-i,to) ){
                pt = ptm[ma*size_d+job];
                remain[ma] -= pt;
                tmp0=max(tmp0,tmp1)+pt;
            }
            if(lane==0)back[_nbMachines-1-i]=tmp0;
        }
    }
}

template <unsigned size>
inline __device__ void
tile_remainingWork(thread_block_tile<size> g, const int *unscheduledJobs, const int nUn, int *remain)
{
    int lane = g.thread_rank();
    int job;
    int ptjob;

    for (int i = 0; i <= nUn / g.size(); i++) {
        for(int j=0;j<_nbMachines;j++){
            ptjob=0;
            if (i * g.size() + lane < nUn) {
                job = unscheduledJobs[i * g.size() + lane];
                ptjob=_tempsJob[j * size_d + job];
            }

            int rem=tile_sum(g,ptjob);
            g.sync();

            if(lane==0)
                remain[j]+=rem;

            g.sync();
        }
    }
}

inline __device__ void
addFrontAndBound(const int* back, const int* front, const int* remain, int job, const int* ptm, int &lowerb)
{
    int tmp0,tmp1;

    int lb=front[0]+remain[0]+back[0];
    tmp0=front[0]+ptm[job];

    for(int j=1;j<_nbMachines;j++){
        tmp1=max(tmp0,front[j]);
        tmp0=tmp1+ptm[j*size_d+job];
        lb=max(lb,tmp1+remain[j]+back[j]);//cmax);
    }
    lowerb=lb;
}

template <typename T>
inline __device__ void
addBackAndBound(const int* back, const int* front, const int* remain, T job, const int* ptm, int &lowerb)
{
    int tmp0,tmp1;

    tmp0=back[(_nbMachines-1)]+ptm[(_nbMachines-1)*size_d+job];
    int lb=front[_nbMachines-1]+remain[_nbMachines-1]+back[_nbMachines-1];
    //add job to back and compute max of machine-bounds;

    for(int j=_nbMachines-2;j>=0;j--){
        tmp1=max(tmp0,back[j]);//+pt;
        tmp0=tmp1+ptm[j*size_d+job];
        lb=max(lb,front[j]+remain[j]+tmp1);
    }
    lowerb=lb;
}

// template <typename T>
template <unsigned size>
inline __device__ void
tile_addFrontAndBound(thread_block_tile<size> g, const int* back, const int* front, const int* remain, const int *unscheduledJobs, const int nUn, int *cost)
{
    for (int i = g.thread_rank(); i < nUn; i+=g.size()) {        // one thread : one job
        int job = unscheduledJobs[i]; // each thread grabs one job
        addFrontAndBound(back, front, remain, job, _tempsJob, cost[job]);
    }
}

template <unsigned size>
inline __device__ void
tile_addBackAndBound(thread_block_tile<size> g, const int* back, const int* front, const int* remain, const int *unscheduledJobs, const int nUn, int *cost)
{
    for (int i = g.thread_rank(); i < nUn; i+=g.size()) {        // one thread : one job
        int job = unscheduledJobs[i]; // each thread grabs one job
        addBackAndBound(back, front, remain, job, _tempsJob, cost[job]);
    }
}

template <unsigned size>
inline __device__ void
tile_resetRemain(thread_block_tile<size> g, int* remain)
{
    for (int i = g.thread_rank(); i < _nbMachines; i+=g.size()) {
        remain[i] = _sumPT[i];
    }
}







inline __device__ int perIVMtodo(int& ivmtodo, int* row, int line)
{
    ivmtodo = 0;
    while(row[ivmtodo]>=0 && ivmtodo < size_d-line)ivmtodo++;

    return ivmtodo;
}

template <typename T>
__global__ void
 __launch_bounds__(128, 8)
boundOne(const T * schedules_d, const T * limit1s_d, const T * limit2s_d, const T * dir, const T * line_d,
  int * costsBE_d, const int * toSwap_d, const int * ivmId_d, const int best, const int * front,
  const int * back)
{
    // thread indexing
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory
    extern __shared__ unsigned char sharedArray[];
    unsigned char * tempsJob_sh = (unsigned char *) sharedArray;

    if (threadIdx.x < size_d) {
        for (int j = 0; j < _nbMachines; j++)
            tempsJob_sh[j * size_d + threadIdx.x] =
              (unsigned char) _tempsJob[j * size_d + threadIdx.x];
    }
    __syncthreads();

//    if(tid>=nbIVM_d)return;
    //
    if (tid < todo) {
        int ivmnb = ivmId_d[tid];                       // the ivm thread tid is working on
        int BE    = dir[index2D(line_d[ivmnb], ivmnb)]; // begin/end?

        int limit1 = limit1s_d[ivmnb] + 1 - BE;
        int limit2 = limit2s_d[ivmnb] - BE;

        int Swap1 = toSwap_d[tid];//index of job to place...
        int Swap2 = (1 - BE) * limit1 + BE * limit2;//...at begin or end

        int jobnb = schedules_d[index2D(Swap1, ivmnb)];

        int where = ivmnb * 2 * size_d + BE * size_d + (int) jobnb;

        if (line_d[ivmnb] < (size_d - 1)) { // boundNodes
            //costsBE_d[where] = computeCostFast(schedules_d, Swap1, Swap2, limit1, limit2, tempsJob_sh, ivmnb, _tabJohnson, front + ivmnb * _nbMachines, back + ivmnb * _nbMachines, best);
            costsBE_d[where] = computeCostFast(schedules_d, Swap1, Swap2, limit1, limit2, tempsJob_sh, ivmnb, _tabJohnson, front + ivmnb * _nbMachines, back + ivmnb * _nbMachines, best);
        }
    }
} // boundOne

template < unsigned tile_size >
__global__ void // __launch_bounds__(128, 16)
boundWeak_BeginEnd(const int *limit1s_d,const int *limit2s_d, const int *line_d, const int *schedules_d, int *costsBE_d, const int *state_d,int *front_d,int *back_d,const int best,int *flagLeaf)
{
    thread_block_tile<tile_size> g = tiled_partition<tile_size>(this_thread_block());

    int ivm = (blockIdx.x * blockDim.x + threadIdx.x) / tile_size; // global ivm id
    int warpID = threadIdx.x / tile_size;

    return;

    // nothing to do
    if (state_d[ivm] == 0) return;

    // SHARED MEMORY
    // 5*nbMachines*(int) per warp
    // 1*nbJob*(int) per warp
    // 4*(int) per warp
    // (20*nbMachines+4*nbJob+16)*(int) per block
    // +nbJob*nbMachines per block (ptm)
    // 20/20 : 1920 B
    // 50/20 :
    extern __shared__ bool sharedSet[];
    int *front = (int*)&sharedSet;//partial schedule begin
    int *back    = (int *)&front[4 * _nbMachines];  // partial schedule end[M]
    int *remain  = (int *)&back[4 * _nbMachines];   // remaining work[M]
    int *prmu = (int *)&remain[4 * _nbMachines];   // schedule[N]

    front += warpID*_nbMachines;
    back += warpID*_nbMachines;
    remain += warpID*_nbMachines;
    prmu += warpID * size_d;

    // load PTM to smem
    //load schedule limits and line to smem
    int line=line_d[ivm];
    int l1=limit1s_d[ivm];
    int l2=limit2s_d[ivm];

    for (int i = g.thread_rank(); i < size_d; i+=g.size()) {
        prmu[i]=schedules_d[ivm*size_d+i];
    }
    //initialize remain
    for (int i = g.thread_rank(); i < _nbMachines; i+=g.size()) {
        remain[i] = _sumPT[i];
    }
    g.sync();

    tile_scheduleFront(g, prmu, l1, _tempsJob, front, remain);
    tile_scheduleBack(g, prmu, l2, _tempsJob, back, remain);
    g.sync();

    tile_addFrontAndBound(g,back,front,remain,&prmu[l1+1],size_d-line,&costsBE_d[2 * ivm * size_d]);
    tile_addBackAndBound(g,back,front,remain,&prmu[l1+1],size_d-line,&costsBE_d[(2 * ivm + 1) * size_d]);

    if(g.thread_rank()==0){
        if (line == size_d - 1) {
            flagLeaf[ivm] = 1;
            atomicInc(&targetNode, UINT_MAX);
        }
    }

    //back to main memory : for Johnson bounding only
    if(_boundMode==2){
        g.sync();
        for (int i = g.thread_rank(); i<_nbMachines; i+=g.size())
        {
            front_d[ivm * _nbMachines + i] = front[i];
            back_d[ivm * _nbMachines + i]  = back[i];
        }
    }
}

template < unsigned tile_size >
__global__ void
boundWeak_Begin(int *limit1s_d,int *limit2s_d,const int *line_d,int *schedules_d,int *costsBE_d, int *state_d,int *front_d,int *back_d,const int best,int *flagLeaf)
{
    thread_block_tile<tile_size> tile32 = tiled_partition<tile_size>(this_thread_block());

    int ivm = (blockIdx.x * blockDim.x + threadIdx.x) / tile_size; // global ivm id
    int warpID = threadIdx.x / tile_size;

    // SHARED MEMORY
    // 5*nbMachines*(int) per warp
    // 1*nbJob*(int) per warp
    // 4*(int) per warp
    // (20*nbMachines+4*nbJob+16)*(int) per block
    // +nbJob*nbMachines per block (ptm)
    // 20/20 : 1920 B
    // 50/20 :
    extern __shared__ bool sharedSet[];
    int *front = (int*)&sharedSet;//partial schedule begin
    int *back    = (int *)&front[4 * _nbMachines];  // partial schedule end[M]
    int *remain  = (int *)&back[4 * _nbMachines];   // remaining work[M]
    int *prmu = (int *)&remain[4 * _nbMachines];   // schedule[N]

    // load PTM to smem
    prmu += warpID * size_d;
    //load schedule limits and line to smem
    int line=line_d[ivm];
    int l1=limit1s_d[ivm];
    int l2=limit2s_d[ivm];

    int i;

	for(i=tile32.thread_rank(); i<size_d; i+=tile_size)
	{
        prmu[i]=schedules_d[ivm*size_d+i];
	}
    //zero remain
    for (i = tile32.thread_rank(); i < _nbMachines; i+=tile_size) {
        remain[warpID * _nbMachines + i] = _sumPT[i];
    }
    tile32.sync();

    // nothing to do
    if (state_d[ivm] == 0) return;

    tile_scheduleFront(tile32, prmu, l1, _tempsJob, &front[warpID * _nbMachines], &remain[warpID*_nbMachines]);
    tile_scheduleBack(tile32, prmu, l2, _tempsJob, &back[warpID * _nbMachines], &remain[warpID*_nbMachines]);

    tile32.sync();

    tile_addFrontAndBound(tile32,&back[warpID * _nbMachines],&front[warpID * _nbMachines],&remain[warpID * _nbMachines],&prmu[l1+1],size_d-line,&costsBE_d[2 * ivm * size_d]);

    if(tile32.thread_rank()==0){
        if (line == size_d - 1) {
            flagLeaf[ivm] = 1;
            atomicInc(&targetNode, UINT_MAX);
        }
    }

    //back to main memory
    tile32.sync();
    for (i = 0; i <= (_nbMachines / tile_size); i++) {
        if (i * tile_size + tile32.thread_rank() < _nbMachines) {
            front_d[ivm * _nbMachines + i * tile_size + tile32.thread_rank()] = front[warpID * _nbMachines + i * tile_size + tile32.thread_rank()];
            back_d[ivm * _nbMachines + i * tile_size + tile32.thread_rank()]  = back[warpID * _nbMachines + i * tile_size + tile32.thread_rank()];
        }
    }
}

//§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
//warp parallel makespan evaluation
//§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
template <unsigned tile_size>
inline __device__ void
tile_evalSolution(thread_block_tile<tile_size> g, const int* permutation, const int limit1, const int* ptm, int *front) {
    int lane=g.thread_rank();

    for(int i=lane;i<_nbMachines;i+=tile_size)
        front[i]=0;

    int slice_width = min(tile_size,limit1+1);
    int n_slices = 1 + limit1/tile_size;

    for(int k=0;k<n_slices;k++){
        int to = min(slice_width,limit1+1-k*slice_width);

        int pt;
        int job;
        int tmp0=0;
        int tmp1=0;
        if(lane<_nbMachines){
            tmp0=front[lane];
            tmp1=tmp0;
        }

        for(int i=0;i<to;i++){ //for jobs scheduled in front
            tmp1=g.shfl_up(tmp0,1); //no wrap-around (lowest lane not modified)
            if(lane<min(i+1,_nbMachines)){
                job=permutation[k*slice_width+i-lane];
                pt = ptm[lane*size_d+job];
                tmp0=max(tmp0,tmp1)+pt;
            }
        }
        if(lane==0)
            front[0]=tmp0;

        if(lane<to){
            job=permutation[to+k*slice_width-lane-1];
        }

        for(int i=1;i<_nbMachines;i++){
            tmp1=g.shfl_down(tmp0,1);
            //top-thread needs to read from memory
            if(k>0 && lane == to-1 && lane+1 < _nbMachines)
                tmp1 = front[lane+i];

            if(lane<min(_nbMachines-i,to)){ // mini(_nbMachines-i,mNM)
                pt = ptm[(lane+i)*size_d+job];
                tmp0=max(tmp0,tmp1)+pt;
            }
            if(lane==0)front[i]=tmp0;
        }
    }
}

template <unsigned tile_size>
__global__ void // __launch_bounds__(128, 16)
makespans(int *schedules_d,int *cmax, int *state_d)
{
    thread_block_tile<tile_size> tile32 = tiled_partition<tile_size>(this_thread_block());

    int ivm = (blockIdx.x * blockDim.x + threadIdx.x) / tile_size; // global ivm id
    int warpID = threadIdx.x / tile_size;

    extern __shared__ char sharedCmax[];
    int *front = (int*)&sharedCmax;//partial schedule begin
    int *prmu = (int *)&front[4 * _nbMachines];   // schedule[N]

    // load PTM to smem
    prmu += warpID * size_d;
    int i;
    for(i=tile32.thread_rank(); i<size_d; i+=tile_size){
        prmu[i]=schedules_d[ivm*size_d+i];
    }
    tile32.sync();

    // nothing to do
    if (state_d[ivm] == 0) return;

    tile_evalSolution(tile32, prmu, _nbJob-1, _tempsJob, &front[warpID * _nbMachines]);
    tile32.sync();

    if(tile32.thread_rank()==0){
        cmax[ivm]=front[(warpID+1) * _nbMachines - 1];
    }
}

//=========================================================
template <unsigned tile_size>
__device__ void tile_2point(thread_block_tile<tile_size> g, int *p1, int *p2, int *c, int *flag, int l1, int l2)
{
	int i;

	for(i=g.thread_rank(); i<size_d; i+=tile_size){
		flag[i]=0;
	}
	g.sync();

	for(i=g.thread_rank(); i<l1+1; i+=tile_size){
		int job=p1[i];
        c[i]=job;
        flag[job]=1;
	}
	g.sync();

	for(i=l2+g.thread_rank(); i<size_d; i+=tile_size){
		int job=p1[i];
 		c[i]=job;
		flag[job]=1;
	}
	g.sync();

	if(g.thread_rank() == 0)
	{
	    int ind=0;
	    for(int i=l1+1;i<l2;i++)
	    {
	        while(flag[p2[ind]]!=0)ind++;
	        c[i]=p2[ind];
	        flag[p2[ind]]=1;
	    }
	}
}

template <unsigned tile_size>
__global__ void // __launch_bounds__(128, 16)
xOver_makespans(int *schedules_d,int *cmax, int *state_d, int *parent2, int* l1, int* l2)
{
    thread_block_tile<tile_size> tile32 = tiled_partition<tile_size>(this_thread_block());

    int ivm = (blockIdx.x * blockDim.x + threadIdx.x) / tile_size; // global ivm id
    int warpID = threadIdx.x / tile_size;

    extern __shared__ char sharedCmax[];
    int *front = (int*)&sharedCmax;//partial schedule begin
    int *prmu = (int *)&front[4 * _nbMachines];   // schedule[N]
    int *chld = (int *)&prmu[4 * size_d];
    int *flag = (int *)&chld[4 * size_d];

    // load PTM to smem
    prmu += warpID * size_d;
    int i;
	for(i=tile32.thread_rank(); i<size_d; i+=tile_size){
        prmu[i]=schedules_d[ivm*size_d+i];
	}
	chld += warpID * size_d;
	flag += warpID * size_d;

    tile32.sync();

    // nothing to do
    if (state_d[ivm] == 0) return;

	tile_2point(tile32, prmu, parent2, chld, flag, l1[ivm], l2[ivm]);
    tile_evalSolution(tile32, chld, _nbJob-1, _tempsJob, &front[warpID * _nbMachines]);
    tile32.sync();

	for(i=tile32.thread_rank(); i<size_d; i+=tile_size){
        schedules_d[ivm*size_d+i]=chld[i];
	}
    if(tile32.thread_rank()==0){
        cmax[ivm]=front[(warpID+1) * _nbMachines - 1];
    }
}



//single block :

__global__ void
__launch_bounds__(1024, 1)
boundRoot(int *mat, int *dir, int *line, int *costsBE_d, int *sums_d, const int best, const int branchingMode) {
	thread_block bl = this_thread_block();
    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
    int warpID = bl.thread_rank() / warpSize;

    extern __shared__ int smem1[];
    int     *permut = smem1;

    for(int l=bl.thread_rank(); l<size_d; l+=bl.size())
    {
        permut[l] = l;
    }
    bl.sync();

    for(int l=bl.thread_rank(); l<size_d; l+=bl.size()){
        // bound begin
        costsBE_d[l] =
        computeCost(permut, 0, l, 0, size_d, _tempsJob, 0, _tabJohnson, 999999);
        // atomicAdd(&sums_d[0], costsBE_d[l]);
    }

    if(branchingMode>0){
        for(int l=bl.thread_rank(); l<size_d; l+=bl.size()){
            // bound end
            costsBE_d[size_d + l] =
            computeCost(permut, size_d - 1, l, -1, size_d - 1, _tempsJob, 0, _tabJohnson, 999999);
            // atomicAdd(&sums_d[1], costsBE_d[size_d + l]);
        }
    }
    bl.sync();

    line[0] = 0;
    dir[index2D(0, 0)] = 0;



    int d=0; //default
    switch(branchingMode){
    case 1:{
        d=tile_branchMaxSum(tile32, mat, costsBE_d, dir, line[0]);
        break;}
    case 2:{
        d=tile_branchMinMin(tile32, mat, costsBE_d, dir, line[0]);
        break;}
    case 3:{
        d=tile_MinBranch(tile32, mat, costsBE_d, dir, line[0],best);
        break;}
    }

    d=tile32.shfl(d,0);
    tile32.sync();//!!!! every thread has dir


    if(bl.thread_rank()==0){
        dir[0] = d;

        //if Bwd reverse job order
        int i1=0;
        int i2=size_d - 1;

        if(d==1){
            while(i1<i2)
            {
                swap_d(&mat[i1],&mat[i2]);
                i1++;
                i2--;
            }
        }
    }
    bl.sync();

    //prune
    for(int l=bl.thread_rank();l<size_d;l+=bl.size()){
        int job=absolute_d(mat[l]);
        int val = costsBE_d[index2D(job, dir[0])];
        if (val >= best) {
            mat[l] = negative_d(job);
        }
    }
    bl.sync();

    for(int l=bl.thread_rank();l<size_d;l+=bl.size()){
        root_d[l] = mat[l];
    }
    if (bl.thread_rank() == 0) root_dir_d = dir[0];

    // if(bl.thread_rank()==0){
    //     for(int i=0;i<size_d;i++){
    //         printf("%d ",costsBE_d[i]);
    //     }
    //     printf("\n");
    //     for(int i=0;i<size_d;i++){
    //         printf("%d ",costsBE_d[size_d+i]);
    //     }
    //     printf("\n");
    //     for(int i=0;i<size_d;i++){
    //         printf("%d ",root_d[i]);
    //     }
    //     printf("\n %d \n",root_dir_d);
    // }
}
