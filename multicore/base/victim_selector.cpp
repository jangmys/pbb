#include <victim_selector.h>

#include <memory>

// namespace victim_selector
// {
std::unique_ptr<VictimSelector> make_victim_selector(const unsigned _nthreads, const char _type)
{
    switch (_type) {
        case 'r':
        {
            return std::make_unique<RingVictimSelector>(_nthreads);
        }
        case 'a':
        {
            return std::make_unique<RandomVictimSelector>(_nthreads);
        }
        case 'o':
        {
            return std::make_unique<HonestVictimSelector>(_nthreads);
        }
    }
    return nullptr;
}
// }
