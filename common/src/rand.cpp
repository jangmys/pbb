#include "rand.hpp"

int
intRand(const int min, const int max)
{
    thread_local std::mt19937 generator(std::random_device{ } ());

    std::uniform_int_distribution<int> distribution(min, max);   // closed [min,max]
        return distribution(generator);
}

float
floatRand(const float min, const float max)
{
    thread_local std::mt19937 generator(std::random_device{ } ());

    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(generator);
}

void
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
