#ifndef INTERVAL_H
#define INTERVAL_H

#include <vector>
#include <memory>

#include "gmp.h"
#include "gmpxx.h"

class interval {
public:
  interval() = default;
  interval(mpz_class _begin, mpz_class _end,int _id) : begin(_begin),end(_end),id(_id){};
  interval(std::string b,std::string e, std::string _id);
  interval(const interval& i);

  mpz_class begin = mpz_class(0);
  mpz_class end = mpz_class(0);
  int id = 0;

  bool operator < (const interval& in) const
  { return (begin < in.begin);};

  mpz_class length() const;

  bool disjoint(interval *i) const;

  void operator=(interval& i);
  bool operator==(interval& i) const;
};

#endif
