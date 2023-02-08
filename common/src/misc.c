#include <assert.h>

#include "misc.h"

inline int
negative(const int i)
{
    return ((i >= 0) ? (-1-i) : i);
}

inline int
absolute(const int i)
{
    return (i >= 0) ? i : (-1-i);
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

void remove_insert_left(int *arr, const int a, const int b)
{
    if(a==b)return;
    assert(b>a);

    int tmp = arr[b];

    for(int i=b;i>a;i--)
        arr[i] = arr[i-1];

    arr[a] = tmp;
}

void remove_insert_right(int *arr, const int a, const int b)
{
    if(a==b)return;
    assert(a<b);

    int tmp = arr[a];

    for(int i=a;i<b;i++)
    {
        arr[i] = arr[i+1];
    }

    arr[b] = tmp;
}
