#include <algorithm>

#include "subproblem.h"
#include "ils.h"
#include "ls.h"

IG::IG(instance_abstract * inst)
{
    neh=new fastNEH(inst);
    nhood=new fspnhood<int>(inst);

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


    ls = new LocalSearch(inst);
    // ls = std::make_unique<LocalSearch>(inst);

}

IG::~IG()
{
	delete nhood;
    delete neh;
}

void IG::shuffle(int *array, int n)
{
    if (n > 1) {
	    for (int i = 0; i < n - 1; i++) {
			int j = helper::intRand(i, n-1);
	        // size_t j = i + drand48() / (RAND_MAX / (n - i) + 1);
	        int t = array[j];
	        array[j] = array[i];
	        array[i] = t;
	    }
    }
}

int IG::makespan(subproblem* s)
{
    return nhood->m->computeHeads(s->schedule.data(), s->size);
}

//randomly removes k elements from perm and stores them in permout
void IG::destruction(int *perm, int *permOut, int k)
{
	std::vector<int>v(nbJob);
	for(int i=0;i<nbJob;i++)v[i]=i;

  	std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v.begin(), v.end(), g);

    std::vector<int>flag(nbJob);
    for(int i=0;i<nbJob;i++)flag[i]=1;

	//remove k jobs : copy to permOut
    for(int i=0;i<k;i++){
        permOut[i]=perm[v[i]];
        flag[v[i]]=0;
    }

	//prefix sum
    std::vector<int>pref(nbJob,0);
    for(int i=1;i<nbJob;i++){
        pref[i]=pref[i-1]+flag[i-1];
    }

    std::vector<int>tmp(nbJob,0);
    for(int i=0;i<nbJob;i++){
        if(flag[i]){
            tmp[pref[i]]=perm[i];
        }
    }

    for(int i=0;i<nbJob;i++){
        perm[i]=tmp[i];
    }
}

void IG::destruction(int *perm, int *permOut, int k, int a, int b)
{
	if(b-a < k){
		std::cout<<a<<" "<<b<<" "<<k<<"destruction not possible\n"; exit(-1);
	}

	std::vector<int>v;

	for(int i=a;i<b;i++)v.push_back(i);

  	std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v.begin(), v.end(), g);

	std::vector<int>flag(nbJob);
    for(int i=0;i<nbJob;i++)flag[i]=1;

    for(int i=0;i<k;i++){
        permOut[i]=perm[v[i]];
		flag[v[i]]=0;
    }

	std::vector<int>pref(nbJob,0);
    for(int i=1;i<nbJob;i++){
        pref[i]=pref[i-1]+flag[i-1];
    }

	std::vector<int>tmp(nbJob,0);
    for(int i=0;i<nbJob;i++){
        if(flag[i]){
            tmp[pref[i]]=perm[i];
        }
    }

    for(int i=0;i<nbJob;i++){
        perm[i]=tmp[i];
    }
}

void IG::perturbation(int *perm, int k, int a, int b)
{
    std::vector<int>sel1(b-a);

	//SELECT k RANDOM
    for(int i=0;i<b-a;i++){
        sel1[i]=a+i;
    }
    shuffle(sel1.data(),(b-a));

    std::vector<int>sel2(k);

    for(int i=0;i<k;i++){
        sel2[i]=sel1[i];
        sel1[i]=perm[sel1[i]];
    }

    shuffle(sel2.data(),k);

    for(int i=0;i<k;i++){
        perm[sel2[i]]=sel1[i];
    }
}


void IG::construction(std::vector<int>& perm, std::vector<int>& permOut, int k)
{
    int cmax;
    int len=nbJob-k;

    for(int j=0;j<k;j++){
        nhood->m->bestInsert(perm.data(), len, permOut[j], cmax);
    }
}

void IG::construction(std::vector<int>& perm, int *permOut, int k,int a, int b)
{
    int cost;
	int len=nbJob-k;

    for(int j=0;j<k;j++){
		nhood->m->bestInsert(perm.data(), len, permOut[j], cost);
    }
}


// void IG::blockConstruction(int *perm, int *permOut, int k)
// {
//     // int cmax;
//     // int cost,bestpos;
//
//     // int len=nbJob-k;
//
//     for(int i=0;i<=nbJob-k;i++){
//         //
//     }
// }






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
    subproblem* temp=new subproblem(nbJob);
    subproblem* best=new subproblem(nbJob);
    subproblem* reduced=new subproblem(nbJob);

    int currentcost=0;
    int bestcost=0;

    *best=*current;

	bestcost=nhood->m->computeHeads(best->schedule.data(), nbJob);

    // bestcost=neh->evalMakespan(best->schedule, nbJob);
	currentcost=bestcost;
	current->ub=bestcost;

    int l1=0;//current->limit1+1;
    int l2=nbJob;//current->limit2;

	// neh->runNEH(current->schedule,currentcost);
	// std::cout<<"cccc "<<currentcost<<std::endl;
	// currentcost=localSearchBRE(current->schedule);

    // return currentcost;
	int tempcost;
    int perturb=destructStrength;
	bool improved=false;

    //
    for(int iter=0;iter<igiter;iter++){
		*temp=*current;
		// perturbation(temp->schedule, perturb, l1, l2);

		destruction(temp->schedule.data(), reduced->schedule.data(), perturb, l1, l2);
		construction(temp->schedule, reduced->schedule.data(), perturb,l1,l2);

        // destruction(temp->schedule, reduced->schedule, perturb);
		// tempcost=localSearchPartial(temp->schedule,nbJob-perturb);


			// std::cout<<*temp<<"\n";

		// tempcost=localSearchBRE(temp->schedule);
        // tempcost=localSearch(temp->schedule,l1,l2);

		// printf(" bbb === %d %d %d\n",tempcost,bestcost,perturb);

		int kmax=(int)sqrt(nbJob);
		tempcost=localSearchKI(temp->schedule.data(),kmax);
		temp->ub=tempcost;

		// printf(" aaa === %d %d %d\n",tempcost,bestcost,perturb);
		// temp->print();

        if(acceptance(tempcost, currentcost, acceptanceParameter)){
			currentcost=tempcost;
		    *current=*temp;
            perturb=destructStrength;
			// printf(" bbb ======= %d\t",currentcost);
			// current->print();
        }else{
            perturb++;
            if(perturb>5)
                perturb=destructStrength;
        }

        if(tempcost < bestcost){
            bestcost=tempcost;
			*best=*temp;
			improved=true;
			// printf("improved!!!!!!!!! %d\n",best->ub);
        }
    }

    // bestcost=neh->evalMakespan(best->schedule, nbJob);
	// current->copy(best);
	// current->cost=bestcost;

	if(improved){
		*current=*best;
		// printf("improved %d\n",best->ub);
	}

    // best->print();

    delete temp;
    delete best;
    delete reduced;

    return current->ub;
}

int IG::runIG(subproblem* current, int l1, int l2)
{
	// std::cout<<"get: "<<*current<<"\n";

    subproblem* temp=new subproblem(nbJob);
    subproblem* best=new subproblem(nbJob);
    subproblem* reduced=new subproblem(nbJob);

    int currentcost=0;
    int bestcost=0;

    *best=*current;

	bestcost=nhood->m->computeHeads(best->schedule.data(), nbJob);
    // bestcost=neh->evalMakespan(best->schedule, nbJob);
	currentcost=bestcost;

    // int l1=current->limit1+1;
    // int l2=current->limit2;

	// neh->runNEH(current->schedule,currentcost);
	// std::cout<<"cccc "<<currentcost<<std::endl;
	// currentcost=localSearchBRE(current->schedule);

    // return currentcost;
	int tempcost;
    int perturb=destructStrength;
    //
    for(int iter=0;iter<igiter;iter++){
		*temp=*current;
		// perturbation(temp->schedule, perturb, l1, l2);

		destruction(temp->schedule.data(), reduced->schedule.data(), perturb, l1, l2);
		construction(temp->schedule, reduced->schedule.data(), perturb,l1,l2);

        // destruction(temp->schedule, reduced->schedule, perturb);
		// tempcost=localSearchPartial(temp->schedule,nbJob-perturb);


			// std::cout<<*temp<<"\n";

		// tempcost=localSearchBRE(temp->schedule);
        tempcost=localSearch(temp->schedule.data(),l1,l2);

		// printf(" bbb === %d %d %d\n",tempcost,bestcost,perturb);

		// int kmax=(int)sqrt(nbJob);
		// tempcost=localSearchKI(temp->schedule,kmax);

		// printf(" aaa === %d %d %d\n",tempcost,bestcost,perturb);
		// temp->print();

        if(acceptance(tempcost, currentcost, acceptanceParameter)){
			currentcost=tempcost;
		    *current=*temp;
            perturb=destructStrength;
			// printf("%d\t",currentcost);
			// current->print();
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

	// std::cout<<"found: "<<*best<<"\n";

    // bestcost=neh->evalMakespan(best->schedule, nbJob);
	// current->copy(best);
	// current->cost=bestcost;

    // best->print();
    // printf("%d\n",bestcost);

    delete temp;
    delete best;
    delete reduced;

    return bestcost;
}

// int IG::runIG(subproblem* current,subproblem* guide)
// {
//     subproblem* temp=new subproblem(nbJob);
//     subproblem* best=new subproblem(nbJob);
//     subproblem* reduced=new subproblem(nbJob);
//
//     int currentcost=0;
//     int bestcost=0;
//
//     // temp->copy(current);
// 	*best=*current;
//     bestcost=neh->evalMakespan(best->schedule, nbJob);
// 	currentcost=bestcost;
//
// 	printf("curr %d\n",currentcost);
//     // current->print();
//
//     int l1=current->limit1+1;
//     int l2=current->limit2;
//
// 	currentcost=ris(current,guide);
// 	// currentcost=localSearchBRE(current->schedule);
//
// 	int tempcost;
//     int perturb=destructStrength;
//     //
//     for(int iter=0;iter<igiter;iter++){
// 		*temp=*current;
//         // temp->copy(current);
//
// 		// perturbation(temp->schedule, perturb, l1, l2);
//
//         destruction(temp->schedule, reduced->schedule, perturb,l1,l2);
// 		tempcost=localSearch(temp->schedule,l1,l2-perturb);
//
//         construction(temp->schedule, reduced->schedule, perturb,l1,l2);
//
// 		// tempcost=localSearchBRE(temp->schedule);
//         // tempcost=localSearch(temp->schedule,l1,l2);
//
// 		// int kmax=(int)sqrt(nbJob);
//
// 		tempcost=ris(temp,guide);
// 		// tempcost=localSearchKI(temp->schedule,kmax);
//
// 		// printf(" aaa === %d %d\n",tempcost,bestcost);
// 		// temp->print();
//
//         if(acceptance(tempcost, currentcost, acceptanceParameter)){
// 			currentcost=tempcost;
// 			*current=*temp;
//
// 			// printf("%d\t",currentcost);
// 			// current->print();
//         }
//         if(tempcost < bestcost){
//             bestcost=tempcost;
// 			*best=*temp;
//         }
//     }
//
//     // bestcost=neh->evalMakespan(best->schedule, nbJob);
// 	*current=*best;
//
//     // best->print();
//     // printf("%d\n",bestcost);
//
//     delete temp;
//     delete best;
//     delete reduced;
//
//     return bestcost;
// }

int
IG::localSearch(int* const arr, int l1, int l2)
{
    std::vector<int>tmp2(nbJob);

    int best=nhood->m->computeHeads(arr, nbJob);

    int c;

    // int depth = sqrt(nbJob);
    int depth = sqrt(l2-l1);

    for(int k=0;k<10000;k++){
        memcpy(tmp2.data(), arr, nbJob*sizeof(int));
        c=nhood->fastkImove(tmp2.data(), depth,l1,l2);

        // if(acceptance(c, best, 0.01)){
        if(c<best){
            best=c;
            memcpy(arr, tmp2.data(), nbJob*sizeof(int));
			continue;
        }else{
            // printf("\t\t==== %d ===%3d\n",k,best);
            break;
        }
    }

    return best;
}

int
IG::localSearchBRE(int *arr, int l1, int l2)
{
    std::vector<int>tmp(nbJob);

    int best=nhood->m->computeHeads(arr, nbJob);

    bool found;
    int c;

    for(int k=0;k<10000;k++){
        found=false;
        for(int i=l1+1;i<l2;i++){
            memcpy(tmp.data(), arr, nbJob*sizeof(int));
            c=nhood->fastBREmove(tmp.data(), i, l1, l2);

            if(c<best){
                found=true;
				best=c;
				memcpy(arr, tmp.data(), nbJob*sizeof(int));
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
IG::localSearchKI(int *arr,const int kmax)
{
    std::vector<int>tmp(nbJob);

    int best=nhood->m->computeHeads(arr, nbJob);

    bool found;
    int c;
	int i;

	//ls iterations ... 10000 = 'infinity' (likely getting trapped in local min much earlier)
    for(int k=0;k<10000;k++){
        found=false;
		//for all neighbors
        for(int j=0;j<nbJob;j++){
			i=visitOrder[j];

            memcpy(tmp.data(), arr, nbJob*sizeof(int));
            c=nhood->kImove(tmp.data(), i, kmax);//fastBREmove(tmp, i);

			//accept first improvement...
            if(c<best){
                found=true;
                best=c;
                memcpy(arr, tmp.data(), nbJob*sizeof(int));
                break; //accept first improvement...
            }
        }
        if(!found){
            break;
        }
    }

    return best;
}




int
IG::localSearchPartial(int *arr,const int N)
{
    std::vector<int>tmp(nbJob);

    int len=N;
    int best=nhood->m->computeHeads(arr, len);

    bool found;
    int c;

	//ls iterations ... 10000 = 'infinity' (likely getting trapped in local min much earlier)
    for(int k=0;k<10000;k++){
        found=false;
		//for all neighbors
        for(int j=0;j<len;j++){
			int i=j;//visitOrder[j];

            memcpy(tmp.data(), arr, len*sizeof(int));

            int rjob=nhood->m->remove(tmp.data(), len, i);
            nhood->m->bestInsert(tmp.data(), len, rjob, c);

			//accept first improvement...
            if(c<best){
                found=true;
                best=c;
                memcpy(arr, tmp.data(), len*sizeof(int));
                break; //accept first improvement...
            }
        }
        if(!found){
            break;
        }
    }

    return best;
}

//======================================================

// int
// IG::vbih(subproblem* current, subproblem* guiding)
// {
// 	subproblem* best=new subproblem(*current);
// 	subproblem* temp=new subproblem(*current);
//     subproblem* reduced=new subproblem(nbJob);
//
//     int bestcost=neh->evalMakespan(best->schedule, nbJob);
// 	int currentcost=bestcost;
//
// 	int min_bsize=2;
// 	int bsize=2;
//
// 	int tempcost;
// 	// int len=nbJob;
//
// 	int kmax=sqrt(nbJob);
// 	currentcost=localSearchKI(current->schedule,kmax);
//
//
//     for(int k=1;k<igiter;k++){
// 	    min_bsize=1;
// 	    bsize=1;
//         while(true)
//         {
//             *temp=*current;
//
//             destruction(temp->schedule, reduced->schedule, bsize);
//
//             // for(int i=0;i<bsize;i++){
//             //     printf("%3d ",reduced->schedule[i]);
//             // }
//             // printf("\t|\t");
//
//             // temp->print();
//             // reduced->print();
//
//             //localsearch on partial schedule
//             localSearchPartial(temp->schedule,nbJob-bsize);
//             // localSearchPartial(reduced->schedule,bsize);
//
//             // temp->print();
//             // reduced->print();
//
//             //insert removed
//             tempcost=nhood->m->bestBlockInsert(temp->schedule,nbJob-bsize,reduced->schedule,bsize);
//
//             // len=nbJob;
//
//             // temp->print();
//
//             //local search on complete solution
//     		// int kmax=(int)sqrt(nbJob);
//     		// tempcost=localSearchKI(temp->schedule,kmax);
//
// 		    // tempcost=localSearchBRE(temp->schedule);
//             // tempcost=localSearch(temp->schedule,0,nbJob);
//
//             tempcost=ris(temp, guiding);
//
//             // printf("%d %d %d\n",tempcost,currentcost,bsize);
//
//             if(acceptance(tempcost, currentcost, acceptanceParameter)){
//                 bsize=min_bsize;
//                 currentcost=tempcost;
// 				*current=*temp;
//                 // printf("%d\t",currentcost);
//                 // current->print();
//             }
//             else{
//                 bsize++;
//                 // break;
//             }
//             if(tempcost < bestcost){
//                 *guiding = *temp;
//                 bestcost=tempcost;
// 				*best=*temp;
//             }
//             if(bsize>3)break;
//         }
//     }
//
//
//     // printf("\n\n\n %d \n",bestcost);
//     // best->print();
//
// 	delete best;
// 	delete temp;
//     delete reduced;
//
//     return bestcost;
// }






// int
// IG::ris(subproblem* current, subproblem* guiding)
// {
// 	subproblem* temp=new subproblem(*current);
//     subproblem* star=new subproblem(*guiding);
//
//     int best=nhood->m->computeHeads(temp->schedule, nbJob);
//
//     int len=nbJob;
//     int cmax=0;
//
//     int i=0;
//     int h=0;
//
//     while(i<nbJob)
//     {
//         *temp=*current;
//
//         int job1,rpos;
//         // do{
//             h = h%nbJob;
//             // h = visitOrder[h];
//             job1=star->schedule[h];//job at position h in guiding solution
//             rpos=temp->locate(job1);//position of this job in current
//             h++;
//         // }while(rpos==h);
//
//         if(rpos<0){printf("notfound\n");exit(-1);}
//
//         int rjob=nhood->m->remove(temp->schedule, len, rpos);
//         nhood->m->bestInsert(temp->schedule, len, rjob, cmax);
//
//         if(cmax<best){
//             *current=*temp;
//             best=cmax;
//             i=1;
//         }else{
//             i++;
//         }
//     }
//
// 	// *current=*temp;
//
//     // printf("current %d\n",best);
// 	// current->print();
//
// 	delete star;
// 	delete temp;
//
//
//     return best;
// }
