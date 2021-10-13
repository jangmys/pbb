#ifndef WORKS_H
#define WORKS_H

//=====================================================
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <list>
#include <map>
#include <stack>
#include <deque>
#include <memory>
#include <sys/time.h>
#include <time.h>

#include "solution.h"

#include "gmp.h"
#include "gmpxx.h"


class solution;
class work;
class pbab;

typedef std::shared_ptr<work> work_ptr;

//works are stored in 3 structures, keeping works sorted according to size,id

//size
struct mpz_compare // Du + grand au + petit
{ bool operator()(mpz_class b1, mpz_class b2) const
  {
      return b1 > b2;
  }
};
typedef std::multimap<mpz_class, std::shared_ptr<work>, mpz_compare> sizes_type;
typedef sizes_type::iterator sizes_iterator;

// struct timet_compare // Du + petit au + grand.
// { bool operator()(time_t t1, time_t t2) const
//   {
// 	  return t1 < t2;
//   }
// };
// typedef std::multimap<time_t, std::shared_ptr<work>, timet_compare> times_type;
// typedef times_type::iterator times_iterator;

//ids
struct id_compare // Du + petit au + grand.
{ bool operator()(int t1, int t2) const
  {
	  return t1 < t2;
  }
};
typedef std::map<int, std::shared_ptr<work>, id_compare> id_type;
typedef id_type::iterator id_iterator;

//unassigned works
typedef std::deque<std::shared_ptr<work>>unassigned_type;

class works
{
public:
    works();
    works(std::string directory, pbab*pbb);

    void init(pbab* _pbb);
    void init_complete(pbab* _pbb);
    void init_complete_split(pbab* _pbb, const int nParts);
    void init_complete_split_lop(pbab* _pbb, const int nParts);
    void init_infile(pbab* _pbb);

    pbab*pbb;

    std::string directory;

    //quelques statistiques
    mpz_class size;//total size

	//more statistics ...
    mpz_class totalNodes;
    mpz_class numDivides;
    mpz_class numIntersects;

    struct timespec startt,endt;

    bool end;

//    int panne;
//    int nouveau;
//    int actif;work

    //ensemble d'intervalles organisés selon
    sizes_type sizes;     // la taille
    // times_type times;     // le mis à jour
    id_type ids;          // l'identité

    unassigned_type unassigned;  //intervals not treated by any worker


    // la solution optimale ainsi que son cout
    // int cout;
    // problem*pr;

    bool isEmpty();
	bool nearEmpty();

    void save();
    void clear();
    void shutdown();

    //gerer l'ensemble id
    void id_insert(std::shared_ptr<work> w);
    void id_delete(std::shared_ptr<work> w);
    // void id_update(std::shared_ptr<work> w);
    std::shared_ptr<work> id_find(const int id);// retrouver un work d'un certain id s'il existe

    //gerer l'ensemble times
    // void times_insert(std::shared_ptr<work> w, bool fault = false);
    // void times_delete(std::shared_ptr<work> w);
    // void times_update(const std::shared_ptr<work> &w);
    // std::shared_ptr<work> times_fault();   // retrouver un intervalle dont le processus traitant est en panne
    // std::shared_ptr<work> times_oldest();  //retrouv le plus vieux intervale

    //gerer l'ensemble sizes
    void sizes_insert(std::shared_ptr<work> w);
    void sizes_delete(std::shared_ptr<work> w);
    void sizes_update(const std::shared_ptr<work> &w);
    std::shared_ptr<work> sizes_big() const; //retrouver le plus grand intervalle
	std::shared_ptr<work> ids_oldest() const;

    //master...
    std::shared_ptr<work> nbs_close(int nb);

  //=========== gerer les intervalles =====================
  std::shared_ptr<work> _update(const std::shared_ptr<work> &w, bool& chgd);        // mettre a jour un intervalle
  std::shared_ptr<work> _fault();                // recuperer un intervalle dans le processus est en panne


  std::shared_ptr<work>acquireNewWork(int max,bool&);

	bool dropWork(std::shared_ptr<work> tmp);

//recuperer la seconde moitie d'un work (set of intervals):
//divide up to max intervals
    std::shared_ptr<work> steal(unsigned int max, bool&);
// recuperer le plus ancen intervalle
//	std::shared_ptr<work> _oldest(int max);
    std::shared_ptr<work> _adopt(int max);

//work&
  bool updateBest(int& _cout, solution*_sol);
  void request(std::shared_ptr<work> w, bool& chgd);
  bool request(std::shared_ptr<work> w);//, int& _cout, problem*_pr);
  void voidrequest(work& w);
  };

  std::ostream& operator<<(std::ostream& stream, works& ws);
  std::istream& operator>>(std::istream& stream, works& ws);

#endif
