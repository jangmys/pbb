#include <assert.h>

#include "../../common/include/pbab.h"
#include "ivm.h"

ivm::ivm(int _size) : size(_size),line(0),
    posVect(_size,0),endVect(_size,0),dirVect(_size,0),jobMat(_size*_size,0),
    node(_size){
    clearInterval();
    posVect[0]=size; //makes interval empty
}

subproblem& ivm::getNode(){
    return node;
}

int ivm::getDepth() const{
    return line;
}

void ivm::setDepth(int _line){
    line = _line;
};

void ivm::incrDepth(){
    line++;
};

void ivm::alignLeft(){
    for (int i = getDepth() + 1; i < size; i++) {
        posVect[i] = 0;
    }
};

void ivm::setPosition(int* pos){
    for(int i=0;i<size;i++){
        posVect[i] = pos[i];
    }
}

void ivm::setPosition(int depth, int pos){
    posVect[depth] = pos;
}

int ivm::getPosition(const int _d) const{
    return posVect[_d];
}

void ivm::setEnd(int* end){
    for(int i=0;i<size;i++){
        endVect[i] = end[i];
    }
}

void ivm::setEnd(int depth,int end){
    endVect[depth] = end;
}

int ivm::getEnd(const int _d) const{
    return endVect[_d];
}

void ivm::setDirection(int depth, int dir){
    dirVect[depth] = dir;
};

void ivm::setDirection(int dir){
    dirVect[getDepth()] = dir;
};

int ivm::getDirection(const int _d) const{
    return dirVect[_d];
}

int ivm::getDirection() const{
    return dirVect[getDepth()];
}

void
ivm::setRow(int k, const int *row){
    for(int i=0;i<size;i++){
        jobMat[k*size + i] = row[i];
    }
};

int*
ivm::getRowPtr(int i){
    return jobMat.data()+i*size;
}

int
ivm::getCell(int i, int j) const{
    return jobMat[i*size+j];
}

/**
*/
void ivm::clearInterval()
{
    std::fill(std::begin(posVect),std::end(posVect),0);
    std::fill(std::begin(endVect),std::end(endVect),0);
    posVect[0]=size;
}

/** getter for an interval in factoradic form
*/
void ivm::getInterval(int* pos, int* end)
{
    memcpy(pos,posVect.data(), size*sizeof(int));
    memcpy(end,endVect.data(), size*sizeof(int));
}

//"prune"
void ivm::goRight()
{
    posVect[line]++;
}

//"backtrack"
void ivm::goUp()
{
    //"go up" with line=0 can occur :
    //pos==end==N!-1 --> beforeEnd == true
    //pos[0]==N
    if(line>0){
        posVect[line--] = 0;
        eliminateCurrent();
    }else{
        goRight();
    }
}

//"branch"
void ivm::goDown()
{
    line++;
    generateLine(line, true);
}

inline int removeFlag(int a)
{
    return (a>=0)?a:(-a-1);
}

void
ivm::generateLine(const int line, const bool explore)
{
    int lineMinus1 = line - 1;
    int column     = posVect[lineMinus1];
    int i = 0;

    for (i = 0; i < column; i++)
        jobMat[line * size + i] = removeFlag(getCell(lineMinus1,i));
    for (i = column; i < size - line; i++)
        jobMat[line * size + i] = removeFlag(getCell(lineMinus1,i+1));

    if (explore) {
        posVect[line] = 0;
        dirVect[line] = 0;
    }
}

bool
ivm::lineEndState() const
{
    return posVect[line] >= (size-line);
}

bool
ivm::isLastLine() const
{
    return line == size - 1;
}

bool
ivm::pruningCellState() const
{
    return getCurrentCell() < 0;
}

bool
ivm::beforeEnd() const
{
    for (int i = 0; i < size; i++) {
        if (posVect[i] == endVect[i]) continue;
        if (posVect[i] < endVect[i]) return true;
        if (posVect[i] > endVect[i]) return false;
    }
    return true;
}

int
ivm::vectorCompare(const int* a,const int* b)
{
    for (int i = 0; i < size; i++) {
        if(a[i]==b[i])continue;
        if(a[i]<b[i])return -1;
        if(a[i]>b[i])return 1;
    }
    return 0;
}



bool ivm::selectNext()
{
    while (beforeEnd()) {
        if (lineEndState()) {
            //backtrack...
            goUp();
            continue;
        } else if (pruningCellState()) {
            goRight();
            continue;
        } else { //if (!IVM->pruningCellState()) {
            goDown();// branch
            return true;
        }
    }
    return false;
}

bool ivm::selectNextIt()
{
    auto v = posVect.begin()+line;
    auto m = jobMat.begin()+line*size+(*v);

    while (beforeEnd()) {
        if (*v >= (size-line)){
            //backtrack...
            if(line>0){
                *v=0;
                v--; line--;
                m=jobMat.begin()+line*size+(*v);
                *m = negative(*m);
            }else{
                (*v)++;
                m++;
            }
            continue;
        } else if (*m < 0) {
            (*v)++; m++;
            continue;
        } else { //if (!IVM->pruningCellState()) {
            auto j1 = jobMat.begin()+line*size;
            auto j2 = j1 + size;

            for(auto i = j1; i != j1 + size - line; i++){
                if(i == m)continue;
                *(j2++) = absolute(*i);
            }
            line++;
            *(++v) = 0;
            return true;
        }
    }
    return false;
}

//reads IVM and sets current subproblem
void
ivm::decodeIVM()
{
    const int d = getDepth();

    int l1=-1;
    int l2=size;

    auto v = posVect.begin();
    auto dir = dirVect.begin();

    //-----go down until row before current (already scheduled)-----
    for (int l = 0; l < d; l++) {
        int job     = getCell(l,*(v++)); //negative here woud be a bug
        if (*(dir++) == 0) {
            node.schedule[++l1] = job;
        } else {
            node.schedule[--l2] = job;
        }
    }
    //-----fill remaining with jobs in current row-----
    auto m = jobMat.begin() + d*size;
    for (int l = 0; l < size - d; l++){
        node.schedule[l1 + 1 + l] = absolute(*(m++));
    }

    node.limit1 = l1;
    node.limit2 = l2;
} // prepareSchedule


void reverse_order(int* jobs, int line, int size)
{
    int i1=0;
    int i2=size-line-1;
    while(i1<i2){
        std::swap(jobs[i1], jobs[i2]);
        i1++; i2--;
    }
}



template<typename T>
void ivm::sortSiblingNodes(std::vector<T> lb,std::vector<T> prio)
{
    int _line=line;

    switch (arguments::sortNodes) {
        case 0:
        {
            int *jm = getRowPtr(_line);
            int prev_dir=(_line>0)?dirVect[_line-1]:0;
            if(prev_dir!=dirVect[_line])
            {
                reverse_order(jm,_line,size);
            }
            if(prev_dir==1 && dirVect[_line]==0){
                for (int l = 0; l < size - _line; l++){
                    node.schedule[node.limit1 + 1 + l] = absolute(jm[l]);
                }
            }
            break;
        }
        case 1://non-decreasing cost1
        {
            int *jm = getRowPtr(_line);
            gnomeSortByKeyInc(jm, lb.data(), 0, size-_line-1);
            break;
        }
        case 2://non-decreasing cost1, break ties by priority (set in chooseChildrenSet)
        {
            int *jm = getRowPtr(_line);
            gnomeSortByKeysInc(jm, lb.data(), prio.data(), 0, size-_line-1);
            break;
        }
        case 3:
        {
            int *jm = getRowPtr(_line);
            gnomeSortByKeyInc(jm, prio.data(), 0, size-_line-1);
            break;
        }
        case 4:
        {
            int *jm = getRowPtr(_line);
            gnomeSortByKeysInc(jm, lb.data(), prio.data(), 0, size-_line-1);
            break;
        }
    }
}

void
ivm::eliminateCurrent()
{
    int pos = getPosition(getDepth());
    jobMat[getDepth() * size + pos] = negative(getCurrentCell());
}

void
ivm::displayVector(int *ptr) const
{
    for(int i=0;i<size;i++){
        printf("%3d ",ptr[i]);
    }
    printf("\n");
    fflush(stdout);
}

void
ivm::displayMatrix() const
{
    for(int i=0;i<size;i++){
        printf("%2d%2s",posVect[i],"| ");
        printf("%2d%2s",dirVect[i],(line==i)?"*|":" |");
        printRow(i);
    }
    printf("\n");
    fflush(stdout);
}

void
ivm::printRow(const int r) const
{
    for(int i=0;i<size;i++){
        printf("%3d ",getCell(r,i));
    }
    printf("\n");
};



// count the number of explorable subtrees
int ivm::countExplorableSubtrees(const int line)
{
    int count = 0;

    // for(int i = firstAvailableSubtree(line); i<= endVector[line]; i++)
    for (int i = posVect[line] + 1; i <= endVect[line]; i++)
        if (getCell(line,i) >= 0) count++;
    return count;
}

// determine the position where to cut the line between the 2 threads
int ivm::cuttingPosition(const int line, const int division)
{
	int expSubtrees = countExplorableSubtrees(line);

    assert(expSubtrees <= (endVect[line] - posVect[line]));

	// victim thread keeps (expSubtrees / division) subtrees plus the one it is
	// already exploring
	int keep = expSubtrees / division;

	// determine where the thief's interval should start
	int pos          = posVect[line] + 1;
	int keptSubtrees = 0;

	while (keptSubtrees < keep) {
		if (getCell(line,pos) >= 0) keptSubtrees++;
		pos++;
	}

    assert(pos > posVect[line]);
    assert(pos <= endVect[line]);

	return pos;
}

bool ivm::intervalValid() const{
    for (int i = 0; i < size; i++) {
        if ( (posVect[i] < 0) || (posVect[i] >= size - i) ) {
            return false;
        }
        if ((endVect[i] < 0) || (endVect[i] >= size - i)) {
            return false;
        }
    }
    return true;
}

template void ivm::sortSiblingNodes<int>(std::vector<int> lb,std::vector<int> prio);
