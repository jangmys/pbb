#include "weights.h"

//0 (N-1)!  2*(N-1)!    3*(N-1)!    ... (N-1)*(N-1)! N*(N-1)!
//...
//24
//6
//2
//1
//0 1   2   3   4   5   ... N-1 N
weights::weights(int _size)
{
    depth[_size]     = 1;
    depth[_size - 1] = 1;
    for (int i = _size - 2, j = 2; i >= 0; i--, j++) {
        depth[i]  = depth[i + 1];
        depth[i] *= j;
    }
    // std::cout<<depth[0]<<"\n";
    for (int i = 0; i <= _size; i++) {
        for (int j = 0; j <= _size; j++) {
            W[i][j] = j * depth[i];
           // std::cout<<W[i][j]<<" ";
        }
       // std::cout<<std::endl;
    }
}
