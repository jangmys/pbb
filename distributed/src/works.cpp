// ===============================================================================================
#include <limits.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <string.h>
#include <list>
#include <map>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>

#include "work.h"
#include "ttime.h"
#include "pbab.h"
#include "arguments.h"
#include "incumbent.h"

#include "log.h"

#include "works.h"
#include "master.h"

// =======================================================
void
works::init_complete(size_t N)
{
    std::shared_ptr<work> w = std::make_shared<work>();
    w->set_id();

    mpz_class a("0");
    mpz_class b("1");
    for (size_t i = 1; i <= N; i++) {
        b *= i;//n factorial....
    }
    b -= 1;//...minus one (needed to avoid segfault on conversion to factoradic).

    (w->Uinterval).emplace_back(std::make_shared<interval>(a, b, 0));
    unassigned.push_back(w);

    FILE_LOG(logINFO)<<"INITIAL WORK UNITS: "<<unassigned.size();
}

void
works::init_complete_split(size_t N, const int nParts)
{
    std::shared_ptr<work> w = std::make_shared<work>();
    w->set_id();

    mpz_class a("0");
    mpz_class b("1");
    for (size_t i = 1; i <= N; i++)
        b *= i;
    b -= 1;

    (w->Uinterval).emplace_back(std::make_shared<interval>(a, b, 0));
    w->split(nParts);

    INTERVAL_IT it = (w->Uinterval).begin();

    for (it = (w->Uinterval).begin(); it != (w->Uinterval).end(); ++it) {
        std::shared_ptr<work> tmp = std::make_shared<work>();
        (tmp->Uinterval).push_back(std::move(*it));

        unassigned.push_back(tmp);
    }

    FILE_LOG(logINFO)<<"INITIAL WORK UNITS: "<<unassigned.size();
} // works::init_complete_split

void
works::init_complete_split_lop(size_t N, const int nParts)
{
    std::shared_ptr<work> w = std::make_shared<work>();
    w->set_id();

    mpz_class a("0");
    mpz_class b("1");
    for (size_t i = 1; i <= N; i++)
        b *= i;
    b -= 1;

    FILE_LOG(logINFO) << "Searching interval: " << a << "\t" << b;

    (w->Uinterval).emplace_back(std::make_shared<interval>(a, b, 0));
    w->split2(nParts);


    INTERVAL_IT it = (w->Uinterval).begin();

    for (it = (w->Uinterval).begin(); it != (w->Uinterval).end(); ++it) {
        std::shared_ptr<work> tmp = std::make_shared<work>();
        (tmp->Uinterval).push_back(std::move(*it));
        unassigned.push_back(tmp);
    }

    std::reverse(unassigned.begin(),unassigned.end());

    FILE_LOG(logINFO)<<"INITIAL WORK UNITS: "<<unassigned.size();
} // works::init_complete_split

void works::clear()
{
    size = 0;
    for (id_iterator i = ids.begin(); i != ids.end(); ++i){
        (i->second)->clear();
    }
    ids.clear(); sizes.clear();
}

// ========================================================================================
void
works::sizes_insert(std::shared_ptr<work> w)
{
    w->set_size();
    //	std::cout << "   " << size << " ";
    size += w->size;
    //	std::cout << "   " << size << std::endl;
    sizes.insert(sizes_type::value_type(w->size, w));
}

// see sizes_update
void
works::sizes_delete(std::shared_ptr<work> w)
{
    std::pair<sizes_iterator, sizes_iterator> range = sizes.equal_range(w->size);
    for (sizes_iterator i = range.first; i != range.second; i++)
        if (i->second == w) {
            size -= w->size;
            sizes.erase(i);
            break;
        }
}

//
void
works::sizes_update(const std::shared_ptr<work> &w)
{
    sizes_delete(w);
    sizes_insert(w);// call set_size
}

mpz_class
works::get_size()
{
    return size;
}



// ===========================================================================================

void
works::id_insert(std::shared_ptr<work> w)
{
    //	ids.insert(sizes_type::value_type(w->id, w));
    ids.insert(id_type::value_type(w->id, w));
}

void
works::id_delete(std::shared_ptr<work> w)
{
    id_iterator i = ids.find(w->id);

    if (i != ids.end()) {
        //		delete (*i).second;
        ids.erase(i);
    }
}

//===============================================================================================
std::shared_ptr<work>
works::id_find(const int _id)
{
    id_iterator tmp = ids.find(_id);

    return (tmp == ids.end()) ? nullptr : tmp->second;
}

std::shared_ptr<work>
works::sizes_big() const
{
    // return largest
    return sizes.begin()->second; //return largest

    //return largest work which has been updated at least once
    //(the idea is to avoid some redundancy - but it may just block the work spread in the beginning)
    // for(auto i:sizes)
    // {
    //     if(i.second->nb_updates > 0)return i.second;
    // }
    // return nullptr;
}


std::shared_ptr<work>
works::ids_oldest() const
{
    //return oldest work which is at least average size
    for(auto i:ids)
    {
        if(i.second->size >= size/ids.size())return i.second;
        // if(i.second->nb_updates > 0)return i.second;
    }
    return nullptr;
}

std::shared_ptr<work>
works::acquireNewWork(int max, bool &tooSmall)
{
    if(isEmpty()){
        return nullptr;
    }else if (!unassigned.empty())   {
        return adopt(max);
    } else  {
        return steal(max, tooSmall);
    }
    // std::cout<<ids.size()<<std::endl;
}

std::shared_ptr<work>
works::steal(unsigned int max, bool &tooSmall)
{
    // select largest work unit (that was at least updated once!)
    std::shared_ptr<work> tmp1 = ids_oldest();
    // std::shared_ptr<work> tmp1 = sizes_big();

    //if sizes_big returned nothing
    if(!tmp1)return nullptr;

    //create new work by division of tmp1
    std::shared_ptr<work> tmp2(std::move(tmp1->divide(max)));

    //no steal
    // if(tmp1)
    //     tmp1->end_updated=1;
    // return nullptr;

    // DUPLICATION (interval too small)
    if (tmp2->isEmpty()) {
        FILE_LOG(logDEBUG)<<"Divide-BIG returned empty";
        // std::cout<<"duplicate\n"<<std::flush;
        // tooSmall = true;
        return nullptr;

        // if ((tmp1->Uinterval).size() <= max) {
        //     tmp1->end_updated = 1;
        //     *tmp2 = *tmp1;
        //     tmp2->set_id();
        // }
    } else  {
        // tmp1 was modified
        tmp1->end_updated = 1;
        tmp2->end_updated = 0;
        sizes_update(tmp1);
        sizes_insert(tmp2);    // insert created work
        id_insert(tmp2);    // insert created work

        // std::cout<<"stole from "<<tmp1->id<<"\n";
        // std::cout<<"new work "<<tmp2->id<<"\n";

        FILE_LOG(logDEBUG) <<"STOLE"<<tmp1->size;//<<std::endl;
        FILE_LOG(logDEBUG) <<"<<< old \n >>> new";
        FILE_LOG(logDEBUG) <<*tmp2;//<<std::endl;
        //            tmp2->split(max);//split into max parts
        // times_insert(tmp2);    // insert created work
    }

    numDivides++; //counter
    return tmp2;// return created work
} // works::steal


// take entire work from list of unattributed works
std::shared_ptr<work>
works::adopt(int max)
{
    std::shared_ptr<work> tmp = unassigned.front();

    // printf("remaining unexplored :\t%d\n",unassigned.size());

    // tmp->displayUinterval();

    int nbinterv = (tmp->Uinterval).size();
    // printf("%d\t%d",nbinterv,max);

    if (nbinterv == 0) {
        // printf("THIS %d\n",unassigned.size());fflush(stdout);
        unassigned.pop_front();
        return tmp;// nullptr;
    }

    // contains less intervals than requested : split
    if (nbinterv <= max) {
        // divide into
        tmp->split(max);

        FILE_LOG(logINFO) << "Take unassigned " << (tmp->Uinterval).size();
        // tmp->displayUinterval();

        tmp->set_id();

        id_insert(tmp);
        sizes_insert(tmp);
        // times_insert(tmp, true);
        unassigned.pop_front();

        FILE_LOG(logDEBUG4) << "#unassigned " << unassigned.size();
        return tmp;
        // contains more intervals than requested :
    } else  {
        // printf("\ttake\t");
        //		std::shared_ptr<work> tmp2(std::move(tmp->divide(max)));
        std::shared_ptr<work> tmp2(std::move(tmp->take(max)));

        tmp2->set_id();

        id_insert(tmp2);
        sizes_insert(tmp2);
        // times_insert(tmp2, true);

        return tmp2;
    }
} // works::adopt

bool
works::isEmpty()
{
    return (ids.empty() && unassigned.empty());
}

bool
works::nearEmpty()
{
    if(!unassigned.empty())return false;

    if(!ids.empty()){
        if(sizes.begin()->second->size< 1e9){
            FILE_LOG(logDEBUG1) << "Almost done. Remaining: "<<ids.size();
            for(auto i:sizes)
            {
                FILE_LOG(logDEBUG1) << i.second->size;
            }

            return true;
        }else{
            return false;
        }
    }else{
        return false;
    }
}
