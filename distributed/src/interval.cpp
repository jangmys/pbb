#include "interval.h"

interval::interval(const interval& i)
{
    this->begin = i.begin;;
    this->end   = i.end;
    this->id    = i.id;
}

interval::interval(std::string b,std::string e, std::string _id){
    this->begin = mpz_class(b);
    this->end = mpz_class(e);
    this->id = atoi(_id.c_str());
}

bool
interval::disjoint(interval * i) const
{
    return ((i->end <= begin) || (end <= i->begin));
}

mpz_class
interval::length() const
{
    return (end - begin);
}

bool
interval::operator == (interval& i) const
{
    return (begin == i.begin && end == i.end && id == i.id);
}

void
interval::operator = (interval& i)
{
    begin = i.begin;
    end   = i.end;
    id    = i.id;
}
