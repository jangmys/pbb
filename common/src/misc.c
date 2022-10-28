#include "misc.h"

inline int
negative(const int i)
{
    return ((i < 0) ? i : ((-i) - 1));
}

inline int
absolute(const int i)
{
    return ((i >= 0) ? i : ((-1) * i - 1));
}

inline void int_swap(int* a, int* b){
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

// this is gnome sort (tested it: on small arrays better than insert, quick, std::, bubble, selection sort !) O(n^2) worst case time, O(1) space :)
inline void
gnomeSortByKeyInc(int * arr, const int * key, const int from, const int to)
{
    int i = from + 1;
    int j = i + 1;

    while (i <= to) {
        if (key[arr[i - 1]] > key[arr[i]]) {
            int_swap(&arr[i - 1], &arr[i]);
            if (--i) continue;
        }
        i = j++;
    }
}

inline void
gnomeSortByKeysInc(int * arr, const int * key1, const int * key2, const int from, const int to)
{
    int i = from + 1;
    int j = i + 1;

    while (i <= to) {
        if ( (key1[arr[i - 1]] > key1[arr[i]]) ||
          ((key1[arr[i - 1]] == key1[arr[i]]) && (key2[arr[i - 1]] > key2[arr[i]])) )
        {
            int_swap(&arr[i - 1], &arr[i]);
            if (--i) continue;
        }
        i = j++;
    }
}
