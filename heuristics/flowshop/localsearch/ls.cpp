#include "ls.h"

LocalSearch::LocalSearch(instance_abstract& _inst)
{
    nhood = std::make_unique<fspnhood<int>>(_inst);
}


int LocalSearch::operator()(std::vector<int>& perm, int l1, int l2)
{
    std::vector<int>tmp(perm);

    int best=nhood->m->computeHeads(perm, perm.size());

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
LocalSearch::localSearchBRE(std::vector<int>& perm, int l1, int l2)
{
    int nbJob = perm.size();

    std::vector<int>tmp(nbJob);

    int best=nhood->m->computeHeads(perm, nbJob);

    for(int k=0;k<10000;k++){
        bool found=false;
        for(int i=l1+1;i<l2;i++){
            memcpy(tmp.data(), perm.data(), nbJob*sizeof(int));
            int c=nhood->fastBREmove(tmp, i, l1, l2);

            if(c<best){
                found=true;
				best=c;
				memcpy(perm.data(), tmp.data(), nbJob*sizeof(int));
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
    int nbJob = perm.size();

    std::vector<int>tmp(nbJob);

    int best=nhood->m->computeHeads(perm, nbJob);

    int c;
	int i;

	//ls iterations ... 10000 = 'infinity' (likely getting trapped in local min much earlier)
    for(int k=0;k<10000;k++){
        bool found=false;
		//for all neighbors
        for(int j=0;j<nbJob;j++){
			i=j;

            memcpy(tmp.data(), perm.data(), nbJob*sizeof(int));
            c=nhood->kImove(tmp, i, kmax);//fastBREmove(tmp, i);

			//accept first improvement...
            if(c<best){
                found=true;
                best=c;
                memcpy(perm.data(), tmp.data(), nbJob*sizeof(int));
                break; //accept first improvement...
            }
        }
        if(!found){
            break;
        }
    }

    return best;
}
