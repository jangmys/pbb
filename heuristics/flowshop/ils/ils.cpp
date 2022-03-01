#include <algorithm>

#include "subproblem.h"
#include "ils.h"
#include "ls.h"

IG::IG(instance_abstract * inst) :
    neh(std::make_unique<fastNEH>(inst)),
    nhood(std::make_unique<fspnhood<int>>(inst)),
    ls(std::make_unique<LocalSearch>(inst))
{
    // std::cout<<"here\n"<<std::endl;

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

int IG::makespan(subproblem* s)
{
    return nhood->m->computeHeads(s->schedule, s->size);
}

void IG::destruction(std::vector<int>& perm, std::vector<int>& permOut, int k, int a, int b)
{
	if(b-a < k){
		std::cout<<a<<" "<<b<<" "<<k<<"destruction not possible\n"; exit(-1);
	}

	std::vector<int>v;

	for(int i=a+1; i<b; i++)
        v.push_back(i);

  	std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(v.begin(), v.end(), g);

    //the postions to remove
    std::vector<int> removePos(v.begin(),v.begin()+k);
    std::sort(removePos.begin(),removePos.end(),std::greater<int>());

    for(int i=0;i<k;i++){
        permOut[i]=perm[removePos[i]];
    }
    for(int i=0;i<k;i++){
        perm.erase(perm.begin()+removePos[i]);
    }

}

void IG::perturbation(int *perm, int k, int a, int b)
{
    std::random_device rd;
    std::mt19937 g(rd());

    //SELECT k RANDOM positions
    std::vector<int>sel1(b-a);
    for(int i=0;i<b-a;i++){
        sel1[i]=a+i;
    }

    std::shuffle(sel1.begin(), sel1.end(), g);
    // shuffle(sel1.data(),(b-a));
    std::vector<int>sel2(k);
    for(int i=0;i<k;i++){
        sel2[i]=sel1[i];
    }

    for(int i=0;i<k;i++){
        sel1[i]=perm[sel2[i]];
    }

    std::shuffle(sel1.begin(), sel1.begin()+k, g);

    for(int i=0;i<k;i++){
        perm[sel2[i]]=sel1[i];
    }
}


// void IG::construction(std::vector<int>& perm, std::vector<int>& permOut, int k)
// {
//     int cmax;
//     int len=nbJob-k;
//
//     for(int j=0;j<k;j++){
//         nhood->m->bestInsert(perm, len, permOut[j], cmax);
//     }
// }

void IG::construction(std::vector<int>& perm, std::vector<int>& permOut, int k,int a, int b)
{
    int cost;
	int len=nbJob-k;

    nhood->m->tabupos->clear();
    for(int i=0;i<=a;i++){
        nhood->m->tabupos->add(i);
    }
    for(int i=b;i<perm.size();i++){
        nhood->m->tabupos->add(i);
    }

    for(int j=0;j<k;j++){
		nhood->m->bestInsert(perm, len, permOut[j], cost);
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

	float r = helper::floatRand(0.0, 1.0);

    float prob=exp(earg);

    if(r<prob)return true;
    else return false;
}

int IG::runIG(subproblem* current)
{
    std::unique_ptr<subproblem> temp = std::make_unique<subproblem>(nbJob);
    std::unique_ptr<subproblem> best = std::make_unique<subproblem>(nbJob);

    // subproblem* temp=new subproblem(nbJob);
    // subproblem* best=new subproblem(nbJob);
    // subproblem* reduced=new subproblem(nbJob);

    int currentcost=0;
    int bestcost=0;

    *best=*current;

	bestcost=nhood->m->computeHeads(best->schedule, nbJob);

    // bestcost=neh->evalMakespan(best->schedule, nbJob);
	currentcost=bestcost;
	current->set_fitness(bestcost);

    int l1=0;//current->limit1+1;
    int l2=nbJob;//current->limit2;

	// neh->runNEH(current->schedule,currentcost);
	// std::cout<<"cccc "<<currentcost<<std::endl;
	// currentcost=localSearchBRE(current->schedule);

    int perturb=destructStrength;
    std::vector<int> removedJobs(perturb);

	bool improved=false;
    int kmax=(int)sqrt(nbJob);

    for(int iter=0;iter<igiter;iter++){
		*temp=*current;

		destruction(temp->schedule, removedJobs, perturb, l1, l2);
		construction(temp->schedule, removedJobs, perturb,l1,l2);

		int tempcost=ls->localSearchKI(temp->schedule,kmax);

		temp->set_fitness(tempcost);

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

    return current->fitness();
}

int IG::runIG(subproblem* current, int l1, int l2)
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

		destruction(temp->schedule, removedJobs, perturb, l1, l2);
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
