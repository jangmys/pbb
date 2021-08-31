#ifndef MISC_H_
#define MISC_H_

#include <stdio.h>
#include <stdlib.h>

#include <random>
// inline prevents "multiple inclusion" (header included in pbab.h, itself included everywhere)

static void
swap(int * ptrA, int * ptrB)
{
    int tmp = *ptrA;

    *ptrA = *ptrB;
    *ptrB = tmp;
}

static int
negative(const int i)
{
    return ((i < 0) ? i : ((-i) - 1));
}

static int
absolute(const int i)
{
    return ((i >= 0) ? i : ((-1) * i - 1));
}

// this is gnome sort (tested it: on small arrays better than insert, quick, std::, bubble, selection sort !) O(n^2) worst case time, O(1) space :)
static void
gnomeSortByKeyInc(int * arr, const int * key, const int from, const int to)
{
    int i = from + 1;
    int j = i + 1;

    while (i <= to) {
        //        if(arr[i-1]<0){printf("BUG1\n");}
        //        if(arr[i]<0){printf("BUG2\n");}

        if (key[arr[i - 1]] > key[arr[i]]) {
            swap(&arr[i - 1], &arr[i]);
            if (--i) continue;
        }
        i = j++;
    }
}

static void
gnomeSortByKeysInc(int * arr, const int * key1, const int * key2, const int from, const int to)
{
    int i = from + 1;
    int j = i + 1;

    while (i <= to) {
        if ( (key1[arr[i - 1]] > key1[arr[i]]) ||
          ((key1[arr[i - 1]] == key1[arr[i]]) && (key2[arr[i - 1]] > key2[arr[i]])) )
        {
            swap(&arr[i - 1], &arr[i]);
            if (--i) continue;
        }
        i = j++;
    }
}

static bool
isSmaller(const int * a, const int * b, int N)
{
    for (int i = 0; i < N; i++) {
        if (a[i] < b[i]) return true;

        if (a[i] > b[i]) return false;
    }
    return true;
}

// static float floatRand(const float min, const float max) {
//     thread_local std::mt19937 generator(std::random_device{}());
//     std::uniform_real_distribution<float> distribution(min, max);
//     return distribution(generator);
// }


namespace helper {
static int
intRand(const int min, const int max)
{
    thread_local std::mt19937 generator(std::random_device{ } ());

    std::uniform_int_distribution<int> distribution(min, max);   // closed [min,max]
    return distribution(generator);
}

static float
floatRand(const float min, const float max)
{
    thread_local std::mt19937 generator(std::random_device{ } ());

    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(generator);
}

static void
shuffle(int * array, size_t n)
{
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            // https://stackoverflow.com/questions/1202687/how-do-i-get-a-specific-range-of-numbers-from-rand
            int j = intRand(i, n - 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}
}


#endif // ifndef MISC_H_
