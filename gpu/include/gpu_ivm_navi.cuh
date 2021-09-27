#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// __________________________________________________________
// true if (fac1 <= fac2)
template <typename T>
inline __device__ bool
isSmaller(const T * fac1, const T * fac2)
{
    #pragma unroll 4
    for (int i = 0; i < size_d; i++) {
        if (fac1[i] < fac2[i]) {
            return true;
        }
        if (fac1[i] > fac2[i]) {
            return false;
        }
    }
    return false;// true;
}

// __________________________________________________________
// true if (posVec <= endVec)
template <typename T>
inline __device__ bool
beforeEnd(const T * posVecs_d, const T * endVecs_d)
{
    #pragma unroll 4
    for (int i = 0; i < size_d; i++) {
        if (posVecs_d[i] < endVecs_d[i]) {
            return true;
        }
        if (posVecs_d[i] > endVecs_d[i]) {
            return false;
        }
    }
    //	return false;
    return true;
}

// true if (posVec <= endVec)
template <typename T>
inline __device__ bool
beforeEndPart(const T * posVecs_d, const T * endVecs_d,const int l)
{
    #pragma unroll 4
    for (int i = l; i < size_d; i++) {
        if (posVecs_d[i] < endVecs_d[i]) {
            return true;
        }
        if (posVecs_d[i] > endVecs_d[i]) {
            return false;
        }
    }
    	return false;
    // return true;
}



//
//
inline __device__ void
parallelGoDown(int * jobMats_d, int * posVecs_d , const int _line, const int lane, const int mid)
{
    int line = _line;

    assert(line >= 0);
    assert(line < size_d - 1);

    int pos = posVecs_d[line];
    assert(pos >= 0);
    assert(pos < size_d);
    //    assert(state==1);
    assert(jobMats_d[line * size_d + pos] >= 0);

    line++;

    int off = 0;
    if (lane >= pos) off = 1;
    if (lane < size_d - line) jobMats_d[line * size_d + lane] = absolute_d(jobMats_d[(line - 1) * size_d + lane + off]);

    posVecs_d[line] = 0;
}

//
//
inline __device__ void
seqGoDown(int * jobMats_d, int * posVecs_d, int * dirVecs_d, const int _line)
{
    int line = _line;

    assert(line >= 0);
    assert(line < size_d - 1);

    int pos = posVecs_d[line];
    assert(pos >= 0);
    assert(pos < size_d);
    //    assert(state==1);
    assert(jobMats_d[line * size_d + pos] >= 0);

    line++;

    for (int i = 0; i < size_d - line; i++) {
        if (i < pos)
            jobMats_d[line * size_d + i] = absolute_d(jobMats_d[(line-1) * size_d + i]);
        else
            jobMats_d[line * size_d + i] = absolute_d(jobMats_d[(line-1) * size_d + i + 1]);
    }

    posVecs_d[line] = 0;
    dirVecs_d[line] = 0;
}


//
//
template <typename T>
inline __device__ void
generateLine2(T * jobMats_d, T * posVecs_d,
  T * dirVecs_d, const T line,
  const T state)
{
    //  int line = line_d[mid];
    assert(line > 0);
    assert(line < size_d);

    int lineMinus1 = line - 1;
    int column     = posVecs_d[lineMinus1];// index2D(lineMinus1, mid)];
    assert(column < size_d);

    int i = 0;
    #pragma unroll 4
    for (i = 0; i < size_d - line; i++) {
        if (i < column) jobMats_d[line * size_d + i] = absolute_d(jobMats_d[lineMinus1 * size_d + i]);
        else jobMats_d[line * size_d + i] = absolute_d(jobMats_d[lineMinus1 * size_d + i + 1]);
        //    if(i<column)jobMats_d[index3D(line,i,mid)] = absolute(jobMats_d[index3D(lineMinus1,i,mid)]);
        //    else jobMats_d[index3D(line,i,mid)] = absolute(jobMats_d[index3D(lineMinus1,i+1,mid)]);
    }

    if (state > 0) {
        posVecs_d[line] = 0; // index2D(line, mid)] = 0;
        dirVecs_d[line] = 0;
    }
}

// ___________________________________________
template <typename T>
inline __device__ void
generateLine(T * jobMats_d, T * posVecs_d,
  T * dirVecs_d, const T * line_d,
  const T * state_d, int mid)
{
    int line = line_d[mid];

    assert(line > 0);
    assert(line < size_d);

    int lineMinus1 = line - 1;
    int column     = posVecs_d[index2D(lineMinus1, mid)];
    assert(column < size_d);

    int i = 0;
    #pragma unroll 4
    for (i = 0; i < size_d - line; i++) {
        if (i < column) jobMats_d[index3D(line, i, mid)] = absolute_d(jobMats_d[index3D(lineMinus1, i, mid)]);
        else jobMats_d[index3D(line, i, mid)] = absolute_d(jobMats_d[index3D(lineMinus1, i + 1, mid)]);
    }

    if (state_d[mid] > 0) {
        posVecs_d[index2D(line, mid)] = 0;
        dirVecs_d[index2D(line, mid)] = 0;
    }
}

// ________________________________________________
template <typename T>
inline __device__ bool
pruningCellState(const T * jobMats_d, const T * posVecs_d, const T * line_d, int mid)
{
    int line = line_d[mid];
    int pos  = posVecs_d[index2D(line, mid)];

    assert(pos >= 0 && pos < size_d);
    assert(line >= 0 && line < size_d);
    return (jobMats_d[index3D(line, pos, mid)] < 0);
}

// ________________________________________________
template <typename T>
inline __device__ void
goUp(T * jobMats_d, T * posVecs_d, T * line_d, int mid)
{
    assert(line_d[mid] != 0);
    //    return; // first line - cannot goUp

    posVecs_d[index2D(line_d[mid], mid)] = 0;
    line_d[mid]--;
    int pos = posVecs_d[index2D(line_d[mid], mid)];
    jobMats_d[index3D(line_d[mid], pos, mid)] = negative(jobMats_d[index3D(line_d[mid], pos, mid)]);
}

// _______________________________________________
template <typename T>
inline __device__ void
goUp2(T * pos, T * lptr, T * jm, T * mat_ptr)
{
    *pos = 0;
    (*lptr)--;
    pos--; // update current pos
    //    mat_ptr = jm + (*lptr) * size_d + (*pos); // update pos in matrix (backtrack)
    //    *mat_ptr = negative_d(*mat_ptr);
}

//                    *pos = 0;
//                  (*lptr)--;                                // aka goUp
//                pos--;                                    // update current pos
//              mat_ptr = jm + (*lptr) * size_d + (*pos); // update pos in matrix (backtrack)
//            *mat_ptr = negative_d(*mat_ptr);
// ________________________________________________
template <typename T>
inline __device__ void
jumpUp(T * pos, T * lptr, T * jm, T * mat_ptr, int cutoff)
{
    // std::shared_ptr<finterval> remain(new finterval(size));
    // memcpy(remain->begin,IVM->posVect,size*sizeof(int));
    // remain->depth_bt=-IVM->line;
    //    remain->cost_cutoff=-IVM->line;

    while ((*lptr) > cutoff) {
        goUp2(pos, lptr, jm, mat_ptr);
    }

    // IVM->line++;
    // bd->prepareSchedule(IVM,arguments::branchingMode);
    // IVM->line--;

    //    IVM->displayVector(bd->schedule);
    //
    //    int c[2];
    //         int c2[2];
    //    bd->bound[SIMPLE]->bornes_calculer(bd->schedule, bd->limit1,bd->limit2, c, 99999);
    //         //         pbb->bound->bornes_calculer(schedule+k*size, limit1[k],limit2[k], c, 99999);
    //         pbb->bound2->bornes_calculer(schedule+k*size, limit1[k],limit2[k], c2, 99999);
    //    printf("%d \n",c[0]);
    //
    //    memcpy(remain->end,IVM->posVect,size*sizeof(int));
    //    for(int i=IVM->line+1;i<size;i++){
    //        remain->end[i]=size-i-1;
    //    }
    //  remain->end[IVM->line]++;

    //    IVM->displayVector(remain->begin);
    //    IVM->displayVector(remain->end);

    //    remain->cost_cutoff=c[0];
    //    remains.push_back(remain);
    //         //
    //         //displayVector(newpos);
    //
    //         //return;
    // }
} // jumpUp

// _______________________________________________
template <typename T>
inline __device__ void
goDown(T * jobMats_d, T * posVecs_d, T * dirVecs_d,
  T * line_d, T * state_d, int mid)
{
    int line = line_d[mid];

    assert(line < size_d - 1);

    int pos = posVecs_d[index2D(line, mid)];
    assert(pos >= 0);
    assert(pos < size_d);
    assert(jobMats_d[index3D(line, pos, mid)] >= 0);

    line_d[mid]++;
    generateLine(jobMats_d, posVecs_d, dirVecs_d, line_d, state_d, mid);
}

// ___________________________________________________________
template <typename T>
inline __device__ void
goRight(T * posVecs_d, const T * line_d, int mid)
{
    assert(posVecs_d[index2D(line_d[mid], mid)] < size_d);
    posVecs_d[index2D(line_d[mid], mid)]++;
}

// ___________________________________________________________
inline __device__ int
lookForJob(int job, const char * array, int begin, int end)
{
    for (int i = (begin < 0 ? 0 : begin); i < end; i++) {
        if (array[i] == job) {
            return i;
        }
    }
    // assert(false);
    return 0;
}

// ___________________________________________________________
template <typename T>
inline __device__ void
funfold_gpu(T * jobMats_sh, T * posVecs_sh,
  T * dirVecs_sh, T * endVecs_sh,
  T * line_sh, T * state_sh, int matId)
{
    if (line_sh[matId] < size_d - 1) {
        line_sh[matId]++;
        generateLine(jobMats_sh, posVecs_sh, dirVecs_sh, line_sh, state_sh, matId);
    } else {
        state_sh[matId] = 1;
    }
}

// ___________________________________________________________
// ___________________________________________________________
// ___________________________________________________________
// template < int T>
inline __device__ void
tile_decodeIVM(thread_block_tile<32> g, const int * jm, const int * pos, const int * dir, const int line, int& l1, int& l2, int * prmu)
{
    int lane = g.thread_rank();

    if (lane == 0) {
        l1 = -1;
        l2 = size_d;

        int pointed, job;

        for (int j = 0; j < line; j++) {
            pointed = pos[j];
            job     = jm[j * size_d + pointed];
            if (dir[j] == 0) {
                prmu[++l1] = job;
            } else {
                prmu[--l2] = job;
            }
        }
    }

    jm += line * size_d; // row of matrix ...
    g.sync();

    for (int i = lane; i < size_d - line; i += g.size()) {
        prmu[l1 + 1 + i] = jm[i]; // unscheduled jobs
    }

    // complete unscheduled jobs (read current row of matrix)...
    // if(dir[line]==dir[line-1]){
    //     for (int i = lane; i < size_d - line; i += g.size()) {
    //         prmu[l1 + 1 + i] = jm[i]; // unscheduled jobs
    //     }
    // }else{
    //     for (int i = lane; i < size_d - line; i += g.size()) {
    //         prmu[l1 + 1 + i] = jm[_nbJob-line-1-i]; // unscheduled jobs
    //     }
    // }
}

// ==================================================
inline __device__ void
thread_decodeIVM(const int * jm, int * pos, const int * dir, int line, int& l1, int& l2, int * prmu, int num)
{
    l1 = -1;
    l2 = size_d;

    int job;

    // "common part"
    for (int j = 0; j < line; j++) {
        job = jm[j * size_d + pos[j]];
        if (dir[j] == 0) {
            prmu[++l1] = job;
        } else {
            prmu[--l2] = job;
        }
    }

    // fill rest
    for (int j = 0; j < size_d-line; j++) {
        prmu[l1+1+j] = absolute_d(jm[line * size_d + j]);
    }

    // for (int j = l1+1; j < l2; j++) {
    //     swap_d(&prmu[j], &prmu[j + pos[j]]);
    // }
}

// ==================================================


// ==================================================


// ==================================================
// ==================================================
template <unsigned tile_size>
inline __device__ void
tile_prune(thread_block_tile<tile_size> g, int * jm_row, const int * costs, const int dir, const int line, const int best)
{
    int lane = g.thread_rank();

    int job, val;

    for (int i = 0; i <= _nbJob / g.size(); i++) {
        if (i * g.size() + lane < size_d - line) { // one thread : one matrix cell in row
            job = absolute_d(jm_row[i * g.size() + lane]);
            val = costs[index2D(job, dir)];
            #ifdef FINDALL
            if (val > best || line == _nbJob - 1) {
            #else
            if (val >= best || line >= _nbJob - 1) {
            #endif
                jm_row[i * g.size() + lane] = negative_d(job);
            }
        }
    }
}

// ==================================================
template <unsigned tile_size>
inline __device__ int
tile_branchMaxSum(thread_block_tile<tile_size> g, const int * jm_row, const int * costs, int * dir, const int line)
{
    int lane = g.thread_rank();
    int job;
    int delta = 0;
    int val   = 0;

    //compute sum(begin-end)
    for (int i = 0; i <= (size_d / tile_size); i++) {
        delta = 0;
        if (i * tile_size + lane < size_d - line) {
            job   = jm_row[i * tile_size + lane];
            delta = costs[job] - costs[size_d + job];
        }
        delta = tile_sum(g, delta);
        if (lane == 0)
            val += delta;
    }
    //val = sum(delta) < 0  <==>  cost(begin) < cost(end)

    if (lane == 0)
        dir[line] = (val < 0) ? 1 : 0;

    g.sync();

    return dir[line];
}

template <unsigned tile_size>
inline __device__ int
tile_branchMinMin(thread_block_tile<tile_size> g, const int * jm_row, const int * costs, int * dir, const int line)
{
    int lane = g.thread_rank();
    int job;
    int minLB=INT_MAX;
    int val;

    //determine minimum LB among both sets
    for (int i = 0; i <= (size_d / g.size()); i++)
    {
        val = INT_MAX;
        if (i * tile_size + lane < size_d - line) {
            job   = jm_row[i * tile_size + lane];
            val = min(costs[job],costs[size_d + job]);
        }
        g.sync();
        val = tile_min(g, val); //min-reduce val : 0 has reduced value
        if (lane == 0)
            minLB = min(minLB,val);
        g.sync();
    }

    minLB=g.shfl(minLB,0); //broadcast
    g.sync();
    //=================================

    int p0 = 0;
    int p1 = 0;

    for (int i = 0; i <= (size_d / g.size()); i++)
    {
        int costBegin = 99999;
        int costEnd = 99999;
        if (i * tile_size + lane < size_d - line) {
            job   = jm_row[i * tile_size + lane];
            costBegin = costs[job];
            costEnd = costs[size_d + job];
        }
            // int nbBegin = g.ballot((minLB == costs[job]));
            // int nbEnd = g.ballot((minLB == costs[size_d + job]));
        p0 += __popc ( g.ballot((minLB == costBegin)) );
        p1 += __popc ( g.ballot((minLB == costEnd)) );
        g.sync();
    }

    // if (lane == 0)
    dir[line] = (p0>p1) ? 1 : 0;

    g.sync();

    return dir[line];
}

template <unsigned tile_size>
inline __device__ int
tile_MinBranch(thread_block_tile<tile_size> g, const int * jm_row, const int * costs, int * dir, const int line, const int ub)
{
    int lane = g.thread_rank();
    int job;

    int p0 = 0;
    int p1 = 0;

    for (int i = 0; i <= (size_d / g.size()); i++)
    {
        int costBegin = 0;
        int costEnd = 0;
        if (i * warpSize + lane < size_d - line) {
            job   = jm_row[i * warpSize + lane];
            costBegin = costs[job];
            costEnd = costs[size_d + job];
        }

        p0 += __popc ( g.ballot((ub > costBegin)) );
        p1 += __popc ( g.ballot((ub > costEnd)) );
        g.sync();
    }

    //p0,p1 = survivors in begin/end

    // if (lane == 0)
    dir[line] = (p0>p1) ? 1 : 0;
    g.sync();

    //break tie
    if(p0 == p1){
        dir[line] = tile_branchMaxSum(g, jm_row, costs, dir, line);
    }

    return dir[line];
}

template <unsigned tile_size>
__device__ void
initStep(thread_block_tile<tile_size> g, int *jm, int *pv, int *dv, int &line, int &state)
{
    int * mat_ptr = jm + line * size_d + *(pv + line);

    if (*mat_ptr < 0) { // aka pruningCellState
        state = 1;
        for(int i=line+1+g.thread_rank();i<size_d;i+=g.size()){
            pv[i] = 0;
        }
    }else{
        if (g.thread_rank() == 0) {
            if (line < size_d - 2) {
                line++;
                generateLine2(jm, pv, dv, line, state);
            } else {
                state = 1;
            }
        } // end sequential
    }
}

template <unsigned tile_size>
__device__ void
exploreStep(thread_block_tile<tile_size> g, int *jm, int *pv, int *ev, int &line, int &state)
{
    int * mat_ptr = jm + line * size_d + *(pv + line);
    if (g.thread_rank() == 0) {
        int l     = 0;          // first split [pos,end]
        while (pv[l] == ev[l] && l < size_d) l++;

        int *pos = pv + line; // current pos
        int end = ev[l];

        state = 0;

        while (pv[l] <= end) {              // approx check for (pos < end ?)
            //END OF LINE
            if (*pos >= (size_d - line)) {
                if (line == 0) break;      // cannot go up -> interval empty
                *pos = 0;
                line--;                                // aka goUp
                pos--;                                    // update current pos
                mat_ptr = jm + line * size_d + (*pos); // update pos in matrix (backtrack)
                *mat_ptr = negative_d(*mat_ptr);
            }
            else if (*mat_ptr < 0) // aka pruningCellState
            {
                assert(pv[line] < size_d);
                (*pos)++;  // aka goRight
                mat_ptr++; // update pos in matrix (next right)
            } else {
                assert(jm[line * size_d + pv[line]] >= 0);
                assert(line < size_d - 1);

                // found a node to bound --- check validity and set flag to "not empty"
                if (beforeEndPart(pv, ev,l)) {
                    atomicInc(&countNodes_d, INT_MAX);//atomic global counter
                    // count[ivm]++;//per IVM counter
                    state = 1;
                }
                break;
            }
        }
    }
}
