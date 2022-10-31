#ifndef UTIL_HPP
#define UTIL_HPP

#include <vector>
#include <algorithm>

namespace util
{
    template<typename key_type>
    void sort_by_key(std::vector<int>& prmu, const std::vector<key_type>& key);
}


//util_impl.h
namespace util
{
    template<typename key_type>
    void sort_by_key(std::vector<int>& prmu, const std::vector<key_type>& key)
    {
        std::sort(prmu.begin(),prmu.end(),
            [&](const key_type a,const key_type b)
            {
                return key[a] > key[b];
            }
        );
    }
}


#endif
