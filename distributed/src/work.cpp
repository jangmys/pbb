#include <pthread.h>
#include "ttime.h"
#include "pbab.h"
#include "solution.h"
#include "log.h"

#include "work.h"
#include "master.h"

//Constructeurs ===================
//===============================================================
work::work(const work& w) // copie exacte
{
    size = w.size;// ?

    id = w.id;
    max_intervals = w.max_intervals;
    exploredNodes = w.exploredNodes;
    end_updated   = w.end_updated;
    nb_updates = w.nb_updates;

    // copy Uinterval
    Uinterval = w.Uinterval;
}

work::work() // constructeur d'work le non alloué
{
    if (!isEmpty()) Uinterval.clear();

    exploredNodes = 0;
    nb_updates = 0;
    end_updated = 0;
    size = 0;
}

work::work(std::istream& stream)
{
    stream >> *this;
}


work::~work(){ }
//========================================================
//=======================================
size_t
work::writeToFile(FILE * bp)
{
    size_t size = 0;
    int err, k = 0;

    //work ID
    err = fwrite(&id, sizeof(int), 1, bp);
    if (!err) { printf("write: id\n"); exit(-1); }
    size += sizeof(int);

    // how many intervals?
    int num_intervals = Uinterval.size();
    err = fwrite(&num_intervals, sizeof(int), 1, bp);
    if (!err) { printf("write: unknown number of intervals\n"); exit(-1); }
    size += sizeof(int);

    // how many intervals MAX?
    err = fwrite(&max_intervals, sizeof(int), 1, bp);
    if (!err) { printf("write: unknown number of intervals\n"); exit(-1); }
    size += sizeof(int);

    // write out intervals
    for (INTERVAL_IT it = Uinterval.begin(); it != Uinterval.end(); ++it) {
        err = fwrite(&(*it)->id, sizeof(int), 1, bp);
        if (!err) {
            printf("write:unidentified interval\n");
            exit(-1);
        } else {
            size += sizeof(int); // id
        }
        size += mpz_out_raw(bp, ((*it)->begin).get_mpz_t());
        size += mpz_out_raw(bp, ((*it)->end).get_mpz_t());

        k++;
    }

    if (k != num_intervals) { printf("write:wrong interval count\n"); exit(-1); }

    return size;
} // work::writeToFile

// brief: read work from buffer

bool
intervalSmaller(const std::shared_ptr<interval> &A, const std::shared_ptr<interval> &B)
{
    return (A->begin < B->begin);
}

void
work::sortIntervals()
{
    std::sort(Uinterval.begin(), Uinterval.end(), intervalSmaller);
}

//====================================
//===========================================================
void
work::clear()
{
    end_updated = 0;
    size        = 0;
    id = 0;

    if (!isEmpty()) Uinterval.clear();
}

void
work::set_size()
{
    size = 0;

    INTERVAL_IT it;
    // sum up the interval lengths
    for (it = Uinterval.begin(); it != Uinterval.end(); ++it) {
        if ((*it)->end > (*it)->begin)
            size += ((*it)->end - (*it)->begin);
    }

    // it=Uinterval.begin();
    // size=mpz_class("3041409320171337804361260816606476884437764156896050000000000000") - (*it)->begin;
}

static int id_generator = 0;
void
work::set_id()
{
    id = (++id_generator);
}

// work operator ==============================================================================
// mpz_class
// mpzmin(const mpz_class i1, const mpz_class i2)
// {
//     return (i1 < i2) ? i1 : i2;
// }

// mpz_class
// mpzmax(const mpz_class i1, const mpz_class i2)
// {
//     return (i2 > i1) ? i2 : i1;
// }

// static int zero = 0;
// static int replace = 0;
// static int inter = 0;

// =========================================================
// intersect w (worker) with this (server) and write to this
// symmetric (except for result, written to this)
// Ex:
// tmp->intersection(w)
// does
// tmp = intersect (w,tmp)
// =========================================================
// === assumes that both interval collections are SORTED ===
// =========================================================
bool
work::intersection(const std::shared_ptr<work>& w)
{
    bool intersected = false;

    INTERVAL_VEC result;// temp
    result.clear();
    result.reserve((w->Uinterval).size());

    INTERVAL_IT it2 = Uinterval.begin();
    INTERVAL_IT it  = it2;
    //for all intervals in second collection
    for (auto const& jt: w->Uinterval) {
        //scan through first collection
        it = it2;// start here! (Uinterval is sorted)
        while (it != Uinterval.end()) {// linear search
            //it is before jt : move on!
            if ((*it)->end < (jt)->begin) { it++; it2++; continue; }
            //it is after jt : stop searching
            if ((*it)->begin > (jt)->end) { break; }
            //equality of intervals : copy + stop searching
            if ((*it) == (jt)) {
                result.emplace_back((*it));
                break;
            }
            //it and jt overlap
            else if (((*it)->end > (jt)->begin) && ((*it)->begin < (jt)->end)){
                intersected = true;
                result.emplace_back(new interval(
                    std::max((*it)->begin, (jt)->begin),
                    std::min((*it)->end, (jt)->end),
                    (jt)->id)
                );

                // there can be at most one interval in copy with non-empty intresection
                break;
            }
            it++;
        }
    }

    Uinterval = std::move(result);

    return intersected;
} // work::intersection

//
std::shared_ptr<work>
work::take(int max){
    // a new work
    std::shared_ptr<work> tmp(new work());

    if(Uinterval.size()<max){printf("impossible\n");return tmp;}

//    tmp->set_id();

    int i=0;
    while(i<max && Uinterval.size()>0){
        (Uinterval.back())->id=i++;
        (tmp->Uinterval).push_back(Uinterval.back());
        // }
        Uinterval.pop_back();
    }
//    insert((tmp->Uinterval).end(),
//                make_move_iterator(Uinterval.begin()),
//                make_move_iterator(Uinterval.at(max-1)));

    return tmp;
}


// divide up to #max intervals (split)
std::shared_ptr<work> work::divide(int max)
{
    std::shared_ptr<work> tmp(new work());     // a new empty work
    tmp->set_id(); //with an ID

    if (isEmpty()) return tmp; //nothing to get

    mpz_class len(0);
    mpz_class lar(362880);
    mpz_class coupe;

    int nb_stolen = 0;

    // loop over all victim intervals
    for (auto const& it: Uinterval){
        if (nb_stolen >= max) return tmp; // continue;

        len = it->length();// end - (*it)->begin;
        if (len < lar) continue; //too small

        if (len > 0) {
            coupe = it->begin + 1 * len / 2; //cut-point
        } else {
            std::cout << "invalid interval\n" << std::flush; continue;
        }

        (tmp->Uinterval).emplace_back(new interval(coupe+1, it->end, nb_stolen));
        it->end = coupe;

        nb_stolen++;
    }

    return tmp;
} // work::divide

/*
 * //return highest id assigned + 1
 * int work::renumber()
 * {
 *  INTERVAL_IT it;
 *
 *  int _id=0;
 *  for(it = Uinterval.begin(); it != Uinterval.end(); ++it){
 *      (*it)->id=_id++;
 *  }
 *  return _id;
 * }
 */
void
work::split(size_t n)
{
    //	std::cout<<Uinterval.size()<<" ";

    size_t nb    = Uinterval.size();
    size_t ratio = n / nb;

    int newid = 0;//interval ID

    if (ratio <= 1){
        for (INTERVAL_IT it = Uinterval.begin(); it != Uinterval.end(); it++) {
            (*it)->id=(newid++);
        }
        return;
    }

    FILE_LOG(logDEBUG1) <<"split " << nb << " intervals into " << ratio;

    mpz_class len(0);
    mpz_class part(0);

    INTERVAL_VEC result;// temp
    result.clear();
    result.reserve(nb*ratio);

    int ind=0;

    for (INTERVAL_IT it = Uinterval.begin(); it != Uinterval.end(); it++) {
        len  = (*it)->length();
        part = len / ratio;
        result.emplace_back(new interval((*it)->begin, (*it)->begin + part, newid++));
        unsigned int i;
        for (i = 1; i < ratio-1; i++) {
            result.emplace_back(new interval((*it)->begin + i * part +1, (*it)->begin + (i + 1) * part, newid++));
        }
        result.emplace_back(new interval((*it)->begin + (ratio-1) * part+1, (*it)->end, newid++));
    }

    Uinterval = std::move(result);

    //std::cout << Uinterval.size() << std::endl;
}


void
work::split2(size_t n)
{
    //	std::cout<<Uinterval.size()<<" ";

    size_t nb    = Uinterval.size();
    size_t ratio = n / nb;

    int newid = 0;//interval ID

    if (ratio <= 1){
        for (INTERVAL_IT it = Uinterval.begin(); it != Uinterval.end(); it++) {
            (*it)->id=(newid++);
        }
        return;
    }

    FILE_LOG(logINFO) <<"split " << nb << " intervals into " << ratio;
    // std::cout << "split " << nb << " (each) into " << ratio << " parts.";

    mpz_class len(0);
    mpz_class part(0);

    INTERVAL_VEC result;// temp
    result.clear();
    result.reserve(nb*ratio);

    int ind=0;

    mpz_class b(0);
    mpz_class e(0);

    //for all intervals to split
    for (INTERVAL_IT it = Uinterval.begin(); it != Uinterval.end(); it++) {
        b=5*(*it)->begin/100+95*(*it)->end/100;
        e=(*it)->end;

        // std::cout<<e-b<<std::endl;

        result.emplace_back(new interval(b, e, newid++));

        unsigned int i;
        for (i = 1; i < ratio-1; i++) {
            e=b;
            b=5*(*it)->begin/100+95*e/100;

            // std::cout<<i<<" "<<e-b<<std::endl;

            //((*it)->begin+e)/2;

            if(e-b<3628800)break;

            result.emplace_back(new interval(b,e, newid++));
        }

        result.emplace_back(new interval((*it)->begin, b, newid++));
        // len  = (*it)->length();
        // part = len / ratio;
    }

    Uinterval = std::move(result);

    //std::cout << Uinterval.size() << std::endl;
}

/*
 * //operators =====================================================================================
 * mpz_class work::wsize()
 * {
 *  mpz_class ret(0);
 *
 *  for(INTERVAL_IT it = Uinterval.begin(); it != Uinterval.end(); ++it){
 *      ret += (*it)->end - (*it)->begin;
 *  }
 *
 *  return ret;
 * }
 *
 *
 * bool work::equal(work& w)
 * {
 *  bool result=true;
 *
 *  if(Uinterval.size() != w.Uinterval.size())result=false;
 *  else if(Uinterval != w.Uinterval)result=false;
 *
 * //	if((begin != w.begin) || (end != w.end))result=false;
 *
 *  return result;
 * }
 *
 * bool work::differ(std::shared_ptr<work>& w)
 * {
 *  //bool result=false;
 *  size_t numintervals = Uinterval.size();
 *
 *  //#intervals differs => this work & work w differ
 *  if(numintervals != w->Uinterval.size()){
 *      //printf("Dsize\n");
 *      //displayUinterval();w.displayUinterval();
 *      return true; //result=true;
 *  }
 *
 *  for(unsigned int i=0; i<numintervals; i++){
 *      std::shared_ptr<interval>tmp = Uinterval.at(i);
 *      std::shared_ptr<interval>tmpW = (w->Uinterval).at(i);
 *
 *      if(tmpW->id != tmp->id){//printf("Did\n");
 *      return true;}
 *      if(tmpW->begin != tmp->begin){//printf("Dbegin\n");displayUinterval();w.displayUinterval();
 *      return true;}
 *      if(tmpW->end != tmp->end){//printf("Dend\n");displayUinterval();w.displayUinterval();displayUinterval();w.displayUinterval();
 *      return true;}
 *  }
 *  //std::cout<<"Dsame"<<std::endl<<std::flush;
 *
 *  return false;
 * }
 */
void
work::operator = (work& w)
{
    end_updated   = w.end_updated;
    max_intervals = w.max_intervals;
    nb_intervals   = w.nb_intervals;
    nb_updates = w.nb_updates;

    // Uinterval.assign((w.Uinterval).begin(),(w.Uinterval).end());
    Uinterval = w.Uinterval;

    exploredNodes = w.exploredNodes;
    size = w.size;

    id = w.id;
    //	pr = w.pr;
}

// checking=======================================================================================
bool
work::isEmpty()
{
    return Uinterval.empty();
}

/*
  * //=======================================================================================
 * //bool work::intervalSmaller(const interval *A, const interval *B)
 * //{
 * //	return (A->begin < B->begin);
 * //}
 *
 * void work_free(work*w)
 * {
 *  delete w;
 * }
 */

void
work::displayUinterval()
{
    INTERVAL_IT it;

//    printf("displ\n");

    if (isEmpty()){
        printf("empty\n");
        return;
    }

    for (auto it: Uinterval) {
        std::cout << "iid\t" << it->id << " : " << it->begin << "\t" << it->end << std::endl;
    }
}

void
work::readHeader(std::istream& stream)
{
    stream >> id; // int
    stream >> end_updated;// int
    stream >> max_intervals;// int
    stream >> exploredNodes;// mpz_class int
    stream >> nb_intervals; // size_t

    if (stream.fail()) printf("fail. unable to read header\n");
}

int
work::readIntervals(std::istream& stream)
{
    if (!isEmpty()) Uinterval.clear();

    std::string bb, ee, iid;
    for (int i = 0; i < nb_intervals; i++) {
        stream >> iid;
        stream >> bb;
        stream >> ee;

        Uinterval.emplace_back(new interval(bb, ee, iid));
    }

    mpz_class checksz(0);//total size
    stream >> checksz;

    return 1;
} // work::readIntervals

void
work::writeHeader(std::ostream& stream) const
{
    stream << id << " ";
    // was updated remotely?
    stream << end_updated << " ";
    // how many intervals can work w hold?
    stream << max_intervals << " ";
    // stats
    stream << exploredNodes << " ";
    // nb_intervals=Uinterval.size();
    stream << Uinterval.size() << "\n";

    //	if(Uinterval.size()>max_intervals){perror("SIZE\n");exit(-1);}

    if (stream.fail()) std::cout << "writeHeader fail\n" << std::flush;
}

void
work::writeIntervals(std::ostream& stream) const
{
    // how many intervals send?
    //	size_t Usize = Uinterval.size();
    //	stream << Usize << " ";

    // ========= body =========
    // intervals : ID, begin, end
    if (Uinterval.size()) {
        std::vector<std::shared_ptr<interval> >::const_iterator it;
        for (it = Uinterval.begin(); it != Uinterval.end(); ++it) {
            stream << (*it)->id << " ";// std::endl;
            stream << (*it)->begin << " ";// std::endl;
            stream << (*it)->end << std::endl;

            if (stream.fail()) std::cout << "writeIntervals fail at\n" << (*it)->id << " " << (*it)->begin << " "
                                         << (*it)->end << "\n" << std::flush;
        }
    }
    stream << size << std::endl;

    if (stream.fail()) std::cout << "writeIntervals fail\n" << std::flush;
}

// write work to stream
std::ostream&
operator << (std::ostream& stream, const work& w)
{
    w.writeHeader(stream);
    w.writeIntervals(stream);

    return stream;
}

// read work from stream
std::istream&
operator >> (std::istream& stream, work& w)
{
    w.clear();

    w.readHeader(stream);
    if (w.readIntervals(stream) < 0) {
        std::cout << "Error in >> operator\n";
        stream.clear();
        std::cout << "buffer\n" << stream.rdbuf() << "\nwork\n" << w << "\t" << w.nb_intervals << std::flush;
    }
    return stream;
}

size_t
work::readFromFile(FILE * bp)
{
    // printf("read\n");

    Uinterval.clear();// std::vector::clear

    size_t size = 0;
    int err;

    mpz_t tmpb;
    mpz_init(tmpb);
    mpz_t tmpe;
    mpz_init(tmpe);

    // id?
    err = fread(&id, sizeof(int), 1, bp);
    if (!err) { printf("read: id\n"); exit(-1); } else { size += sizeof(int); }

    // how many intervals?
    err = fread(&nb_intervals, sizeof(int), 1, bp);
    if (!err) { printf("read: unknown number of intervals\n"); exit(-1); } else { size += sizeof(int); }

//    printf("reserved for %d \n",nb_intervals);
    if(nb_intervals != 0){
//        printf("RESERVE %d\n",nb_intervals);
        Uinterval.reserve(nb_intervals);
    }

    // how many MAX?
    err = fread(&max_intervals, sizeof(int), 1, bp);
    if (!err) { printf("read: unknown number of intervals\n"); exit(-1); } else { size += sizeof(int); }

//    printf("max %d \n",max_intervals);

    if(nb_intervals==0)return size;

    //    for(int i=0;i<16384;i++){
    int iid;
    for (int i = 0; i < nb_intervals; i++) {
        fread(&iid, sizeof(int), 1, bp);
        if (!err) { printf("read: interval id not read\n"); exit(-1); } else { size += sizeof(int); }

        size += mpz_inp_raw(tmpb, bp);
        size += mpz_inp_raw(tmpe, bp);

        Uinterval.emplace_back(new interval(mpz_class(tmpb), mpz_class(tmpe), iid));
    }

    return size;
} // work::readFromFile
