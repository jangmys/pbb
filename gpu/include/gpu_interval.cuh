/*
 *  author: jan gmys
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__device__ int
countExplorableSubtrees(int l, const int * posVec, const int * endVec, const int * jobMat)
{
    int count = 0;

    for (int i = posVec[l] + 1; i <= endVec[l]; i++)
        if (jobMat[l * size_d + i] >= 0) count++;
    return count;
}

template <typename T>
__device__ int
cuttingPosition(int l, int divi, const T * posVec, const T * endVec, const T * jobMat)
{
    // int nbSubtrees  = endVec[l] - posVec[l];//countAvailableSubtrees(l);
    int expSubtrees = countExplorableSubtrees(l, posVec, endVec, jobMat);

    int keep = expSubtrees / divi;

    if (divi == 1 && expSubtrees > 0) keep = expSubtrees - 1;

    int pos = posVec[l] + 1;
    int keptSubtrees = 0;

    while (keptSubtrees < keep) {
        if (jobMat[l * size_d + pos] >= 0) keptSubtrees++;
        pos++;
    }

    assert(pos > posVec[l]);

    return pos;
}

// =====================================================
inline __device__ int
firstSplit(const int * fact1, const int * fact2)
{
    int ind = 0;

    while (fact1[ind] == fact2[ind])
        ind++;

    return ind;
}

// return position of first non-zero value in fact[id]
inline __device__ int
firstNonZero(const int* fact)
{
    int ind = 0;

    while (fact[ind] == 0)
        ind++;

    return ind;
}

// ________________________________________
// res = floor[a/t] (factoradic/integer)
inline __device__ void
divideVec(int * res, const int * a, const int t)
{
    int r = 0;

    for (int i = 0; i < size_d; i++) {
        int val = a[i] + r * (size_d - i);
        res[i] = val / t;
        r      = val % t;
    }
}
// in-place
inline __device__ void
divideVec(int * a, const int t)
{
    int r = 0;

    for (int i = 0; i < size_d; i++) {
        int val = a[i] + r * (size_d - i);
        a[i]   = val / t;
        r      = val % t;
    }
}

// _______________________________________
// res = a + T.b (factoradic + integer*factoradic)
inline __device__ void
factSaxpy(int * res, const int * a, const int * b, const int T)
{
    int carry = 0;

    for (int i = size_d - 1; i >= 0; i--) {
        int val = a[i] + T * b[i] + carry;
        res[i] = val % (size_d - i);
        carry  = val / (size_d - i);
    }

    //safeguard : if >= (size)! set to MAX
    if (carry > 0) {
        for (int i = size_d - 1; i >= 0; i--)
            res[i] = size_d - i - 1;
    }
}

// _______________________________________
// *end - *pos (subtract factoradic vectors) = *length (no negative results allowed)
inline __device__ void
computeLen(int * length, const int * pos, const int * end)
{
    int hold = 0;

    if (beforeEnd(pos, end)) {
        for (int i = size_d - 1; i >= 0; i--) {
            int d = end[i] - pos[i] - hold;
            if (d >= 0) {
                length[i] = d;
                hold      = 0;
            } else {
                length[i] = d + (size_d - i);
                hold      = 1;
            }
        }
    }
}
// -------------------------
inline __device__ void
getMidpoint(const int * pos, const int * end, int * midvec)
{
    int carry = 0;

    for (int i = 0; i < size_d; i++) {
        midvec[i] = (pos[i] + end[i] + carry * (size_d - i)) / 2;
        carry     = (pos[i] + end[i] + carry * (size_d - i)) % 2;
    }
    for (int i = size_d - 1; i > 0; i--) {
        if (midvec[i] > (size_d - 1 - i)) {
            midvec[i - 1] += midvec[i] / (size_d - i);
            midvec[i]     %= (size_d - i);
        }
    }
}

// ----------------------
inline __device__ void
get1T(const int * pos, const int * end, int * vecC, int den)
{
    int carry = 0;

    // C = ((T-1)*A + B)/T (combi cvx avec \alpha=1/T)
    for (int i = 0; i < size_d; i++) {
        vecC[i] = ((den - 1) * pos[i] + end[i] + carry * (size_d - i)) / den;
        carry   = ((den - 1) * pos[i] + end[i] + carry * (size_d - i)) % den;
    }
    // ajust to valid factoradic
    for (int i = size_d - 1; i > 0; i--) {
        if (vecC[i] > (size_d - 1 - i)) {
            vecC[i - 1] += vecC[i] / (size_d - i);
            vecC[i]     %= (size_d - i);
        }
    }
}

// ----------------------
inline __device__ void
computeAverage_gpu(const int * pos, const int * end, int * average_d, int denominator)
{
    get1T(pos, end, average_d, denominator);
}
