#include <algorithm>

#include "subproblem.h"
#include "ils.h"
#include "ls.h"

IG::IG(const std::vector<std::vector<int>> p_times, const int N, const int M) :
    nhood(std::make_unique<fspnhood<int>>(p_times,N,M)),
    ls(std::make_unique<LocalSearch>(p_times,N,M))
{
    nbJob=nhood->m->nbJob;
    nbMachines=nhood->m->nbMachines;

    int sum=0;
    for(int j=0;j<nbMachines;j++){
        for(int i=0;i<nbJob;i++){
            sum += nhood->m->PTM[j][i];
        }
    }
    avgPT = (float)sum/(nbJob*nbMachines);
    acceptanceParameter=0.2;
	destructStrength=2;

    igiter=200;

    visitOrder = std::vector<int>(nbJob);

    // std::cout<<"heeeree\n"<<std::endl;

	int start=nbJob/2;
	int ind=0;
	for(int i=start;i<nbJob;i++){
		visitOrder[ind]=i;
		ind+=2;
	}
	ind=1;
	for(int i=start-1;i>=0;i--){
		visitOrder[ind]=i;
		ind+=2;
	}
};


IG::IG(instance_abstract& inst) :
    nhood(std::make_unique<fspnhood<int>>(inst)),
    ls(std::make_unique<LocalSearch>(inst))
{
    nbJob=nhood->m->nbJob;
    nbMachines=nhood->m->nbMachines;

    int sum=0;
    for(int j=0;j<nbMachines;j++){
        for(int i=0;i<nbJob;i++){
            sum += nhood->m->PTM[j][i];
        }
    }
    avgPT = (float)sum/(nbJob*nbMachines);
    acceptanceParameter=0.2;
	destructStrength=2;

    igiter=200;

    visitOrder = std::vector<int>(nbJob);

    // std::cout<<"heeeree\n"<<std::endl;

	int start=nbJob/2;
	int ind=0;
	for(int i=start;i<nbJob;i++){
		visitOrder[ind]=i;
		ind+=2;
	}
	ind=1;
	for(int i=start-1;i>=0;i--){
		visitOrder[ind]=i;
		ind+=2;
	}
}


/*
1. choose randomly without repetion k jobs in perm (between positions a+1 and b-1)
2. remove those k jobs
3. return list of removed jobs (and modify sequence perm)
*/
std::vector<int> IG::destruction(std::vector<int>& perm, int k, int a, int b)
{
	if(b-a < k){
		std::cout<<"can't remove "<<k<<" jobs in "<<a+1<<" ... "<<b-1<<". EXIT\n"; exit(-1);
	}

    //indices [a+1 ... b-1]
	std::vector<int>v;
	for(int i=a+1; i<b; i++)
        v.push_back(i);

    //shuffle indices
  	std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v.begin(), v.end(), g);

    //first k indices = the postions to remove
    std::vector<int> removePos(v.begin(),v.begin()+k);
    std::sort(removePos.begin(),removePos.end(),std::greater<int>());
    std::vector<int>removedJobs;

    //save removed jobs
    for(int i=0;i<k;i++){
        removedJobs.push_back(perm[removePos[i]]);
    }
    //remove from permutation
    for(int i=0;i<k;i++){
        perm.erase(perm.begin()+removePos[i]);
    }

    return removedJobs;
}

std::vector<int> IG::destruction(std::vector<int>& perm, int k)
{
    return destruction(perm, k, -1, nbJob);
}






//=====================================================================
// void IG::perturbation(int *perm, int k, int a, int b)
// {
//     std::random_device rd;
//     std::mt19937 g(rd());
//
//     //SELECT k RANDOM positions
//     std::vector<int>sel1(b-a);
//     for(int i=0;i<b-a;i++){
//         sel1[i]=a+i;
//     }
//
//     std::shuffle(sel1.begin(), sel1.end(), g);
//     // shuffle(sel1.data(),(b-a));
//     std::vector<int>sel2(k);
//     for(int i=0;i<k;i++){
//         sel2[i]=sel1[i];
//     }
//
//     for(int i=0;i<k;i++){
//         sel1[i]=perm[sel2[i]];
//     }
//
//     std::shuffle(sel1.begin(), sel1.begin()+k, g);
//
//     for(int i=0;i<k;i++){
//         perm[sel2[i]]=sel1[i];
//     }
// }


//successively insert jobs in permOut in perm (don't touch 0...a, b...N-1)

void IG::construction(std::vector<int>& perm, std::vector<int>& permOut, int k,int a, int b)
{
    int cost;

    nhood->m->tabupos->clear();
    for(int i=0;i<=a;i++){
        nhood->m->tabupos->add(i);
    }
    for(unsigned int i=b;i<perm.size();i++){
        nhood->m->tabupos->add(i);
    }

    for(int j=0;j<k;j++){
		nhood->m->bestInsert(perm, permOut[j], cost);
    }
}

void IG::construction(std::vector<int>& perm, std::vector<int>& permOut)
{
    int cost;

    nhood->m->tabupos->clear();

    for(size_t j=0;j<permOut.size();j++){
        nhood->m->bestInsert(perm, permOut[j], cost);
    }
}






bool IG::acceptance(int tempcost, int cost,float param)
{
    if(tempcost < cost){
        // printf("%d %d ",tempcost,cost);
        return true;
    }

    float earg=(float)cost-(float)tempcost;
    earg /= (param*avgPT/10.0);

	float r = floatRand(0.0, 1.0);

    float prob=exp(earg);

    if(r<prob)return true;
    else return false;
}




void IG::run(std::shared_ptr<subproblem> s)
{
    int makespan = runIG(s,igiter);

    s->ub = makespan;
}



//iterated local search from starting point "current"
int IG::runIG(std::shared_ptr<subproblem> current, const int niter)
{
    std::unique_ptr<subproblem> temp = std::make_unique<subproblem>(nbJob);
    std::unique_ptr<subproblem> best = std::make_unique<subproblem>(nbJob);

    *best=*current;

    int bestcost=nhood->m->computeHeads(best->schedule, nbJob);
    int currentcost=bestcost;
	current->ub=bestcost;

    int perturb=destructStrength;
    std::vector<int> removedJobs(perturb);

	bool improved=false;
    int kmax=(int)sqrt(nbJob);

    for(int iter=0;iter<igiter;iter++){
		*temp=*current;

		removedJobs = destruction(temp->schedule, perturb);
		construction(temp->schedule, removedJobs);

		int tempcost=ls->localSearchKI(temp->schedule,kmax);

		temp->ub = tempcost;

        if(acceptance(tempcost, currentcost, acceptanceParameter)){
			currentcost=tempcost;
		    *current=*temp;
            perturb=destructStrength;
        }else{
            perturb++;
            if(perturb>5)
                perturb=destructStrength;
        }

        if(tempcost < bestcost){
            bestcost=tempcost;
			*best=*temp;
			improved=true;
        }
    }

	if(improved){
		*current=*best;
	}

    return current->ub;
}

int IG::runIG(subproblem* current, int l1, int l2, const int niter)
{
    std::unique_ptr<subproblem> temp = std::make_unique<subproblem>(nbJob);
    std::unique_ptr<subproblem> best = std::make_unique<subproblem>(nbJob);

    int currentcost=0;
    int bestcost=0;

    *best=*current;

	bestcost=nhood->m->computeHeads(best->schedule, nbJob);
	currentcost=bestcost;

    int perturb=destructStrength;
    std::vector<int> removedJobs(perturb);
    //
    for(int iter=0;iter<igiter;iter++){
		*temp=*current;

		removedJobs = destruction(temp->schedule, perturb, l1, l2);
		construction(temp->schedule, removedJobs, perturb,l1,l2);
    //
        int tempcost=(*ls)(temp->schedule,l1,l2);
    //
        if(acceptance(tempcost, currentcost, acceptanceParameter)){
			currentcost=tempcost;
		    *current=*temp;
            perturb=destructStrength;
        }else{
            perturb++;
            if(perturb>5)
                perturb=destructStrength;
        }
        if(tempcost < bestcost){
            bestcost=tempcost;
			*best=*temp;
        }
    }

    return bestcost;
}
