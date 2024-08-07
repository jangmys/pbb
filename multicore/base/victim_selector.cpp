#include <victim_selector.h>

#include <memory>

std::shared_ptr<VictimSelector> make_victim_selector(const unsigned _nthreads, const char _type)
{
    std::cout<<" === Work Stealing : ";

    switch (_type) {
        case 'r':
        {
            std::cout<<" Round-Robin\n";
            return std::make_shared<RingVictimSelector>(_nthreads);
        }
        case 'a':
        {
            std::cout<<" Random\n";
            return std::make_shared<RandomVictimSelector>(_nthreads);
        }
        case 'o':
        {
            std::cout<<" Honest\n";
            return std::make_shared<HonestVictimSelector>(_nthreads);
        }
    }
    return nullptr;
}
// }
