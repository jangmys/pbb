#include "ls.h"

LocalSearch::LocalSearch(instance_abstract& _inst) : nhood(std::make_unique<fspnhood<int>>(_inst))
{
};

LocalSearch::LocalSearch(const std::vector<std::vector<int>> p_times,const int N, const int M) : nhood(std::make_unique<fspnhood<int>>(p_times,N,M))
{
};



int LocalSearch::operator()(std::vector<int>& perm, int l1, int l2)
{
    std::vector<int>tmp(perm);

    int best=nhood->m->computeHeads(perm);

    // int depth = sqrt(nbJob);
    int depth = sqrt(l2-l1);

    for(int k=0;k<1000;k++){
        tmp = perm;

        int c=nhood->fastkImove(tmp, depth,l1,l2);

        if(c<best){
            best=c;
            perm = tmp;
            continue;
        }else{
            break;
        }
    }

    return best;
}


int
LocalSearch::localSearchBRE(std::vector<int>& perm)
{
    std::vector<int>tmp(perm.size());

    int best=nhood->m->computeHeads(perm);

    for(int k=0;k<10000;k++){ //max iterations
        bool found=false;
        for(unsigned i=0;i<perm.size();i++){ //1D moves
            memcpy(tmp.data(), perm.data(), perm.size()*sizeof(int));

            int c=nhood->fastBREmove(tmp, i);
            if(c<best){ //accept move
                found=true;

				best=c;
				memcpy(perm.data(), tmp.data(), perm.size()*sizeof(int));
                break;
            }
        }
        if(!found){
            break;
        }
    }

    return best;
}

int
LocalSearch::localSearchBRE(std::vector<int>& perm, int l1, int l2)
{
    std::vector<int>tmp(perm.size());

    int best=nhood->m->computeHeads(perm);

    for(int k=0;k<10000;k++){
        bool found=false;
        for(int i=l1+1;i<l2;i++){ //only move between limits
            memcpy(tmp.data(), perm.data(), perm.size()*sizeof(int));
            int c=nhood->fastBREmove(tmp, i, l1, l2);

            if(c<best){
                found=true;
				best=c;
				memcpy(perm.data(), tmp.data(), perm.size()*sizeof(int));
                break;
            }
        }
        if(!found){
            break;
        }
    }

    return best;
}


int
LocalSearch::localSearchKI(std::vector<int>& perm,const int kmax)
{
    std::vector<int>tmp(perm.size());

    int best=nhood->m->computeHeads(perm);
    std::cout<<"best : "<<best<<"\n";

	//ls iterations ... 10000 = 'infinity' (likely getting trapped in local min much earlier)
    for(int k=0;k<10000;k++){
        bool found=false;
		//for all neighbors
        for(unsigned j=0;j<perm.size();j++){
			int i=j;

            memcpy(tmp.data(), perm.data(), perm.size()*sizeof(int));
            int c=nhood->kImove(tmp, i, kmax);

			//accept first improvement...
            if(c<best){
                found=true;
                best=c;
                memcpy(perm.data(), tmp.data(), perm.size()*sizeof(int));
                break; //accept first improvement...
            }
        }
        if(!found){
            break;
        }
    }

    return best;
}
