#include "log.h"
#include "pbab.h"
#include "solution.h"
#include "sequentialbb.h"

sequentialbb::sequentialbb(pbab *_pbb, int _size)
{
    size=_size;

    IVM = new ivm(size);
    bd = new ivm_bound<int>(_pbb);

    count_iters=0;
    count_decomposed=0;
    count_lbs=0;
}

sequentialbb::~sequentialbb()
{
    delete IVM;
    delete bd;
}

void
sequentialbb::clear()
{
    IVM->clearInterval();
}

void
sequentialbb::setRoot(const int *varOrder)
{
    clear();

    for(int i=0; i<size; i++){
        IVM->jobMat[i] = varOrder[i];
    }
    IVM->line=0;

    bd->prepareSchedule(IVM);
    bd->boundRoot(IVM);
}

bool
sequentialbb::solvedAtRoot()
{
    bool solved=true;
    for(int i=0;i<size;i++){
        solved &= (IVM->jobMat[i]<0);
    }
    if(solved){
        printf("problem solved at level 0\n");
        for(int i=0; i<size; i++){
            std::cout<<IVM->jobMat[i]<<" ";
        }
        std::cout<<std::endl;
        // IVM->posVect[0]=size;
    }
    return solved;
}

//not thread-safe (setRoot)
void
sequentialbb::initFullInterval()
{
    int * zeroFact = (int *) malloc(size * sizeof(int));
    int * endFact  = (int *) malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        zeroFact[i] = 0;
        endFact[i]  = size - i - 1;
    }

    initAtInterval(zeroFact, endFact);

    free(zeroFact);
    free(endFact);
}

bool
sequentialbb::initAtInterval(int * pos, int * end)
{
    int l = 0;
    IVM->line = l;

    for (int i = 0; i < size; i++) {
        IVM->posVect[i] = pos[i];
        IVM->endVect[i] = end[i];
    }

    if (IVM->beforeEnd()) {
        unfold(arguments::boundMode);
        return true;
    }else{
        return false;
    }
}

void sequentialbb::setBest(const int bestCost)
{
    bd->prune->local_best = bestCost;
}

bool sequentialbb::next()
{
    int state = 0;

    count_iters++;

    /*this loop decomposes one node, if possible*/
    while (IVM->beforeEnd()) {
        if (IVM->lineEndState()) {
            //backtrack...
            IVM->goUp();
            continue;
        } else if (IVM->pruningCellState()) {
            IVM->goRight();
            continue;
        } else if (!IVM->pruningCellState()) {
            state = 1;// exploring

            count_decomposed++;

            IVM->goDown();// branch

            // decode IVM -> subproblems
            bd->prepareSchedule(IVM);

            if (IVM->isLastLine()) {
                bool foundNew=bd->boundLeaf(IVM);

                state = 0;
                continue;
            }
            break;
        }
    }

    if(state == 1)
    {
        switch (arguments::boundMode) {
            case 0:
                bd->weakBoundPrune(IVM);
                break;
            case 1:
                bd->strongBoundPrune(IVM);
                break;
            case 2:
                bd->mixedBoundPrune(IVM);
                break;
        }

        if (IVM->line >= size - 1) {
            printf("too deeep\n");
            exit(0);
        }
    }

    return (state == 1);
}

void
sequentialbb::unfold(int mode)
{
    for (int i = 0; i < size; i++) {
        if ((IVM->posVect[i] < 0) || (IVM->posVect[i] >= size - i)) {
            std::cout << " incorrect position vector " << i << " " << IVM->posVect[i] << std::endl;
            // displayVector(posVect);
            exit(-1);
        }
        if ((IVM->endVect[i] < 0) || (IVM->endVect[i] >= size - i)) {
            std::cout << " incorrect end vector " << i << " " << IVM->endVect[i] << std::endl;
            std::cout << " pos " << i << " " << IVM->posVect[i] << std::endl;
            // displayVector(endVect);
            exit(-1);
        }
    }


    while (IVM->line < size - 2) {
        if (IVM->pruningCellState()) {
            for (int i = IVM->line + 1; i < size; i++) {
                IVM->posVect[i] = 0;
            }
            break;
        }

        IVM->line++;
        IVM->generateLine(IVM->line, false);

        bd->prepareSchedule(IVM);

        for (int i = 0; i < size-IVM->line; i++) {
            int job = IVM->jobMat[IVM->line*size+i];
            if(job<0 || job>=size){
                printf("UNFOLD:invalid job %d (line %d)\n",job,IVM->line);

                IVM->displayVector(IVM->posVect);
                IVM->displayVector(IVM->endVect);
                IVM->displayMatrix();//Vector(jm);
                exit(-1);
            }
        }

        switch (mode) {
            case 0:
                bd->weakBoundPrune(IVM);
                break;
            case 1:
                bd->strongBoundPrune(IVM);
                break;
            case 2:
                bd->mixedBoundPrune(IVM);
                break;
        }
    }
} // matrix::unfold
