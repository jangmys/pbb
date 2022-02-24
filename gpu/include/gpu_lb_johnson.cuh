//CUDA implementation of 2-machine lower bound for PFSP
//(based on Johnson's rule)

inline __device__ void
initCmax_d(const int * tempsMachines, const int nbAffectDebut, int &tmp0, int &tmp1, int &ma0, int &ma1, const int ind)
{
    ma0 = _machine[ind];
    ma1 = _machine[_sum + ind];

    int coeff = __cosf(nbAffectDebut);
    tmp0 = (1 - coeff) * tempsMachines[ma0] + coeff * _minTempsArr[ma0];
    tmp1 = (1 - coeff) * tempsMachines[ma1] + coeff * _minTempsArr[ma1];
}

// §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
inline __device__ void
heuristiqueCmax_d(const int * job, int &tmp0, int &tmp1, const int ma0, const int ma1, const int ind,
  const int * _tabJohnson, const unsigned char * tempsJob)
{
    register int jobCour;

    // #pragma unroll 5
    for (int j = 0; j < size_d; j++) {
        // jobCour = tex1Dfetch(tabJohnson_tex,ind*size_d+j);
        jobCour = _tabJohnson[ind * size_d + j];
        if (job[jobCour] == 0) {
            tmp0 += tempsJob[ma0 * size_d + jobCour];

            tmp1 = max(tmp1, tmp0 + _tempsLag[ind * size_d + jobCour]) + tempsJob[ma1 * size_d + jobCour];
        }
    }
}

inline __device__ int
cmaxFin_d(const int * tempsMachinesFin, const int tmp0, const int tmp1, const int ma0, const int ma1)
{
    return max(tmp1 + tempsMachinesFin[ma1],
             tmp0 + tempsMachinesFin[ma0]);
}

// §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
// compute front
template <typename T>
inline __device__ void
set_tempsMachines_d(const int * front, int * tempsMachines, const T * permutation, const T toSwap1, const T toSwap2,
  const int limit1, const int * tempsJob, const int matId)
{
    int job, m = 0;

    for (m = 0; m < _nbMachines; m++)
        tempsMachines[m] = front[m];

    if (toSwap2 == limit1) {
        job = permutation[index2D(toSwap1, matId)];

        tempsMachines[0] += tempsJob[job];

        for (m = 1; m < _nbMachines; m++) {
            tempsMachines[m]  = max(tempsMachines[m], tempsMachines[m - 1]);
            tempsMachines[m] += tempsJob[m * size_d + job];
        }
    }
}

template <typename T>
inline __device__ void
set_tempsMachines_retardDebut_d(int * tempsMachines, const T * permutation, const T toSwap1, const T toSwap2,
  const int limit1, const int * tempsJob, const int matId)
{
    int job, m = 0;

    memset(tempsMachines, 0, _nbMachines * sizeof(int));

    for (int j = 0; j <= limit1; j++) {
        if (j == toSwap1)
            job = permutation[index2D(toSwap2, matId)];
        else if (j == toSwap2)
            job = permutation[index2D(toSwap1, matId)];
        else
            job = permutation[index2D(j, matId)];

        tempsMachines[0] = tempsMachines[0] + tempsJob[job]; // =_tempsJob[0][job]

        for (m = 1; m < _nbMachines; m++)
            tempsMachines[m] = max(tempsMachines[m], tempsMachines[m - 1]) + tempsJob[m * size_d + job];
    }
}

// §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§!
template <typename T>
inline __device__ void
set_job_d(int * job, const T * permutation, const T toSwap1, const T toSwap2, const int limit1, const int limit2,
  const int matId)
{
    int j = 0;

    for (j = 0; j < size_d; j++) {
        job[j] = 1;
    }

    for (j = limit1 + 1; j < limit2; j++) {
        if (j == toSwap1)
            job[permutation[index2D(toSwap2, matId)]] = 0;
        else if (j == toSwap2)
            job[permutation[index2D(toSwap1, matId)]] = 0;
        else
            job[permutation[index2D(j, matId)]] = 0;
    }
}

template <typename T>
inline __device__ void
set_job_jobFin_d(int * job, int * jobFin, const T * permutation, const T toSwap1, const T toSwap2, const int limit1,
  const int limit2, const int matId)
{
    int j = 0;

    for (j = 0; j <= limit1; j++) {
        if (j == toSwap1)
            job[permutation[index2D(toSwap2, matId)]] = j + 1;
        else if (j == toSwap2)
            job[permutation[index2D(toSwap1, matId)]] = j + 1;
        else
            job[permutation[index2D(j, matId)]] = j + 1;
    }
    for (j = limit1 + 1; j < limit2; j++) {
        if (j == toSwap1)
            job[permutation[index2D(toSwap2, matId)]] = 0;
        else if (j == toSwap2)
            job[permutation[index2D(toSwap1, matId)]] = 0;
        else
            job[permutation[index2D(j, matId)]] = 0;
    }
    for (j = limit2; j < size_d; j++) {
        if (j == toSwap1) {
            job[permutation[index2D(toSwap2, matId)]] = j + 1;
            jobFin[j] = permutation[index2D(toSwap2, matId)];
        } else if (j == toSwap2) {
            job[permutation[index2D(toSwap1, matId)]] = j + 1;
            jobFin[j] = permutation[index2D(toSwap1, matId)];
        } else {
            job[permutation[index2D(j, matId)]] = j + 1;
            jobFin[j] = permutation[index2D(j, matId)];
        }
    }
}

// §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§!
template <typename T>
inline __device__ void
set_tempsMachinesFin_d(const int * back, const T * prmu, int * tempsMachinesFin, const T swap1, const T swap2,
  const int limit2, const int * tempsJob)
{
    int jobCour = 0;

    for (int m = 0; m < _nbMachines; m++)
        tempsMachinesFin[m] = back[m];

    if (swap2 == limit2) {
        jobCour = prmu[swap1];
        tempsMachinesFin[_nbMachines - 1] += tempsJob[(_nbMachines - 1) * size_d + jobCour];
        for (int j = _nbMachines - 2; j >= 0; j--) {
            tempsMachinesFin[j] = max(tempsMachinesFin[j], tempsMachinesFin[j + 1]) + tempsJob[j * size_d + jobCour];
        }
    }
}

inline __device__ void
set_tempsMachinesFin_tempsJobFin_d(const int * jobFin, int * tempsMachinesFin, const int nbAffectFin,
  const int * tempsJob)
{
    int jobCour        = 0;
    int tmpMa[MAXMACH] = { 0 };

    //  #pragma unroll 5
    for (int j = 0; j < _nbMachines; j++) {
        for (int k = j; k < _nbMachines; k++)
            tmpMa[k] = 0;
        for (int k = size_d - nbAffectFin; k < size_d; k++) {
            jobCour   = jobFin[k];
            tmpMa[j] += tempsJob[j * size_d + jobCour];
            for (int l = j + 1; l < _nbMachines; l++) {
                tmpMa[l]  = max(tmpMa[l - 1], tmpMa[l]);
                tmpMa[l] += tempsJob[l * size_d + jobCour];
            }
        }
        tempsMachinesFin[j] = tmpMa[_nbMachines - 1];
    }
}

/**
 * most time intesive part of lower bounding operation.....optimized
 */
// §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
inline __device__ int
borneInfMakespan_d(const int * job, const int * tempsMachinesFin, const int *  tempsMachines, const int nbAffectDebut,
  const int nbAffectFin, int * valBorneInf, const int * tabJohnson, const int * tempsJob, const int best)
{
    int moinsBon = 0;
    int i;

    int ma0  = 0;
    int ma1  = 0;
    int tmp0 = 0;
    int tmp1 = 0;

    int job1;
    int4 job4;

    for (i = 0; i < _sum; i++) {
        initCmax_d(tempsMachines, nbAffectDebut, tmp0, tmp1, ma0, ma1, i);

        // ....manual unrolling!
        // compute johnson seq for ma0,ma1
        for (int j = 0; j < size_d; j += 5) {
            job4.x = _tabJohnson[i * size_d + j];
            job4.y = _tabJohnson[i * size_d + j + 1];
            job4.z = _tabJohnson[i * size_d + j + 2];// vec type...
            job4.w = _tabJohnson[i * size_d + j + 3];
            job1   = _tabJohnson[i * size_d + j + 4];

            if (job[job4.x] == 0) {
                tmp0 += tempsJob[ma0 * size_d + job4.x];
                tmp1  = max(tmp1, tmp0 + _tempsLag[i * size_d + job4.x]);
                tmp1 += tempsJob[ma1 * size_d + job4.x];
            }
            if (job[job4.y] == 0) {
                tmp0 += tempsJob[ma0 * size_d + job4.y];

                tmp1  = max(tmp1, tmp0 + _tempsLag[i * size_d + job4.y]);
                tmp1 += tempsJob[ma1 * size_d + job4.y];
            }
            if (job[job4.z] == 0) {
                tmp0 += tempsJob[ma0 * size_d + job4.z];

                tmp1  = max(tmp1, tmp0 + _tempsLag[i * size_d + job4.z]);
                tmp1 += tempsJob[ma1 * size_d + job4.z];
            }
            if (job[job4.w] == 0) {
                tmp0 += tempsJob[ma0 * size_d + job4.w];

                tmp1  = max(tmp1, tmp0 + _tempsLag[i * size_d + job4.w]);
                tmp1 += tempsJob[ma1 * size_d + job4.w];
            }
            if (job[job1] == 0) {
                tmp0 += tempsJob[ma0 * size_d + job1];

                tmp1  = max(tmp1, tmp0 + _tempsLag[i * size_d + job1]);
                tmp1 += tempsJob[ma1 * size_d + job1];
            }
        }

        if (nbAffectFin != 0) {
            tmp1 = max(tmp1 + tempsMachinesFin[ma1],
                tmp0 + tempsMachinesFin[ma0]);
        } else {
            tmp1 += _minTempsDep[ma1];
        }

        moinsBon = max(moinsBon, tmp1);
    }

    valBorneInf[0] = moinsBon;

    return 0;// moinsBon;
} // borneInfMakespan_d

// §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
inline __device__ int
calculBorne_d(const int * job, const int *  tempsMachinesFin,
  const int *  tempsMachines, const int nbAffectDebut,
  const int nbAffectFin,
  const int * tempsJob, const int * _tabJohnson, const int thebest)
{
    // int minCmax = 0 ;
    int valBorneInf[2] = { 0, 0 };

    // int retardNonFin = 0;//retardNonAff;
    //  int thebest=999999;

    borneInfMakespan_d(job, tempsMachinesFin, tempsMachines, nbAffectDebut,
      nbAffectFin, valBorneInf, _tabJohnson, _tempsJob, thebest);
    return valBorneInf[0];
}
