/*
====================================================
Collection of work units (set of sets of intervals)
--------------------------------------------------------
Author : Jan Gmys (jan.gmys@univ-lille.fr)
------------------------------------------
*/
#ifndef WORKS_H
#define WORKS_H

// =====================================================
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <deque>
#include <memory>

#include "gmp.h"
#include "gmpxx.h"

#include "work.h"

typedef std::shared_ptr<work> work_ptr;

// works are stored in 2 structures, keeping works sorted according to size,id
//==============================================================================
// size
struct mpz_compare // Du + grand au + petit
{ bool
  operator () (mpz_class b1, mpz_class b2) const
  {
      return b1 > b2;
  }
};
typedef std::multimap<mpz_class, std::shared_ptr<work>, mpz_compare> sizes_type;
typedef sizes_type::iterator sizes_iterator;
// =============================================================================
// ids
struct id_compare // Du + petit au + grand.
{ bool
  operator () (int t1, int t2) const
  {
      return t1 < t2;
  }
};
typedef std::map<int, std::shared_ptr<work>, id_compare> id_type;
typedef id_type::iterator id_iterator;
// =============================================================================

// unassigned works
typedef std::deque<std::shared_ptr<work> > unassigned_type;



class works
{
    // ensemble d'intervalles organisés selon
    sizes_type sizes; // la taille
    id_type ids; // l'identité
    unassigned_type unassigned; // intervals not treated by any worker

    mpz_class size;// total size
public:
    works() = default;
    // works(std::string directory, pbab * pbb);

    size_t get_num_works(){return ids.size();}
    size_t get_num_unassigned(){return unassigned.size();}
    bool has_unassigned(){return !unassigned.empty();}

    // init works
    // ================================================
    void init_complete(size_t N);
    void init_complete_split(size_t N, const int nParts);
    void init_complete_split_lop(size_t N, const int nParts);

    // more statistics ...
    mpz_class numDivides;
    mpz_class numIntersects;

    bool
    isEmpty();
    bool
    nearEmpty();

    void
    clear();
    void
    shutdown();

    // gerer l'ensemble id
    // =================================================
    void
    id_insert(std::shared_ptr<work> w);
    void
    id_delete(std::shared_ptr<work> w);
    // void id_update(std::shared_ptr<work> w);
    std::shared_ptr<work>
    id_find(const int id);                      // retrouver un work d'un certain id s'il existe

    std::shared_ptr<work>
    ids_oldest() const;

    // gerer l'ensemble sizes
    // =================================================
    void sizes_insert(std::shared_ptr<work> w);
    void sizes_delete(std::shared_ptr<work> w);
    void sizes_update(const std::shared_ptr<work> &w);
    mpz_class get_size();
    std::shared_ptr<work> sizes_big() const; // retrouver le plus grand intervalle


    // work unit creation
    // ============================================
    std::shared_ptr<work> acquireNewWork(int max, bool&);

    // recuperer la seconde moitie d'un work (set of intervals):
    // divide up to max intervals
    std::shared_ptr<work> steal(unsigned int max, bool&);
    std::shared_ptr<work> adopt(int max);

    // I/O
    // =======================================================
    friend std::ostream&
    operator << (std::ostream& stream, works& ws)
    {
        //number of works
        stream << ws.ids.size() + ws.unassigned.size() << std::endl;

        for (id_iterator i = ws.ids.begin(); i != ws.ids.end(); ++i)
            stream << *(i->second);
        for (auto i: ws.unassigned)
            stream << *i;

        return stream;
    }

    friend std::istream& operator>>(std::istream& stream, works& ws)
    {
        while(!ws.unassigned.empty())ws.unassigned.pop_front();

        int number;
        stream >> number;

        for (int i = 0; i < number; i++) {
            std::shared_ptr<work> w(new work(stream));
            w->set_id();
            w->max_intervals=99999;
            // std::cout<<"work read from stream\n"<<(*w)<<std::endl;
            ws.unassigned.push_back(std::move(w));
        }
        return stream;
    }
};


#endif // ifndef WORKS_H
