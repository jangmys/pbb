#include "../../common/include/pbab.h"
#include "ivm.h"

ivm::ivm(int _size) : size(_size){
    jobMat = (int*)calloc(size*size,sizeof(int));
    posVect = (int*)calloc(size,sizeof(int));
    endVect = (int*)calloc(size,sizeof(int));
    dirVect = (int*)calloc(size,sizeof(int));

    node = std::make_unique<subproblem>(size);

    // posix_memalign((void **) &jobMat, ALIGN, size * size * sizeof(int));
    // posix_memalign((void **) &posVect, ALIGN, size * sizeof(int));
    // posix_memalign((void **) &endVect, ALIGN, size * sizeof(int));
    // posix_memalign((void **) &dirVect, ALIGN, size * sizeof(int));

    clearInterval();

    // line=0;
    posVect[0]=size; //makes interval empty

    // avgline=0.0;
}

ivm::~ivm()
{
    free(jobMat);
    free(posVect);
    free(endVect);
    free(dirVect);
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

int ivm::getDirection(const int _d) const{
    return dirVect[_d];
}

/**
* \brief TODO
*/
void ivm::clearInterval()
{
    memset(jobMat, 0, size*size*sizeof(int));
    memset(posVect, 0, size*sizeof(int));
    memset(endVect, 0, size*sizeof(int));
    posVect[0]=size;
}

/**
* \brief getter for an interval in factoradic form
*/
void ivm::getInterval(int* pos, int* end)
{
    memcpy(pos,posVect, size*sizeof(int));
    memcpy(end,endVect, size*sizeof(int));
}


void ivm::initEmpty()
{
    memset(dirVect, -1, size*sizeof(int));
    memset(posVect, 0, size*sizeof(int));
    memset(endVect, 0, size*sizeof(int));
    memset(jobMat, 0, size*size*sizeof(int));

    line=0;
    for(int i=0;i<size;i++){
        jobMat[i]=i;
    }
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
        posVect[line] = 0;
        line--;
        int pos = posVect[line];
        jobMat[line*size + pos] = negative(jobMat[line*size + pos]);
    }else{
        posVect[line]++;
    }
}

//"branch"
void ivm::goDown()
{
    line++;
    generateLine(line, true);
}

int removeFlag(int a)
{
    if(a>=0){
        return a;
    }else{
        return -a-1;
    }
}

void
ivm::generateLine(const int line, const bool explore)
{
    int lineMinus1 = line - 1;
    int column     = posVect[lineMinus1];
    int i = 0;

    for (i = 0; i < column; i++)
        jobMat[line * size + i] = removeFlag(jobMat[lineMinus1 * size + i]);
    for (i = column; i < size - line; i++)
        jobMat[line * size + i] = removeFlag(jobMat[lineMinus1 * size + i + 1]);

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
	int pos = posVect[line];

    return jobMat[line * size + pos] < 0;
}

bool
ivm::beforeEnd() const
{
    for (int i = 0; i < size; i++) {
        if (posVect[i] < endVect[i]) return true;
        if (posVect[i] > endVect[i]) return false;
    }
    return true;
}





//reads IVM and sets current subproblem
void
ivm::decodeIVM()
{
    const int* const jM  = jobMat;
    const int* const pV  = posVect;
    int _line = line;

    node->limit1 = -1;
    node->limit2 = size;

    for (int l = 0; l < _line; l++) {
        int pointed = pV[l];
        int job     = absolute(jM[l * size + pointed]);

        if (dirVect[l] == 0) {
            node->schedule[++node->limit1] = job;
        } else {
            node->schedule[--node->limit2] = job;
        }
    }
    for (int l = 0; l < size - _line; l++){
        node->schedule[node->limit1 + 1 + l] = absolute(jM[_line * size + l]);
    }
} // prepareSchedule

template<typename T>
void ivm::sortSiblingNodes(std::vector<T> lb,std::vector<T> prio)
{
    int _line=line;

    switch (arguments::sortNodes) {
        case 0:
        {
            int *jm = jobMat + _line * size;
            int prev_dir=(_line>0)?dirVect[_line-1]:0;
            if(prev_dir!=dirVect[_line])
            {
                // std::cout<<"line "<<_line<<" dir "<<IVM->dirVect[_line]<<" reverse\n";
                int i1=0;
                int i2=size-_line-1;
                while(i1<i2){
                    swap(&jm[i1], &jm[i2]);
                    i1++; i2--;
                }
            }
            if(prev_dir==1 && dirVect[_line]==0){
                for (int l = 0; l < size - _line; l++){
                    node->schedule[node->limit1 + 1 + l] = absolute(jm[l]);
                }
            }
            break;
        }
        case 1://non-decreasing cost1
        {
            int *jm = jobMat + _line * size;
            gnomeSortByKeyInc(jm, lb.data(), 0, size-_line-1);
            break;
        }
        case 2://non-decreasing cost1, break ties by priority (set in chooseChildrenSet)
        {
            int *jm = jobMat + _line * size;
            gnomeSortByKeysInc(jm, lb.data(), prio.data(), 0, size-_line-1);
            break;
        }
        case 3:
        {
            int *jm = jobMat + _line * size;
            gnomeSortByKeyInc(jm, prio.data(), 0, size-_line-1);
            break;
        }
        case 4:
        {
            int *jm = jobMat + _line * size;
            gnomeSortByKeysInc(jm, lb.data(), prio.data(), 0, size-_line-1);
            break;
        }
    }
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
        for(int j=0;j<size;j++){
            printf("%3d ",jobMat[i*size+j]);
        }
        printf("\n");
    }
    printf("\n");
    fflush(stdout);
}

// count the number of explorable subtrees
int ivm::countExplorableSubtrees(const int line)
{
    int count = 0;

    // for(int i = firstAvailableSubtree(line); i<= endVector[line]; i++)
    for (int i = posVect[line] + 1; i <= endVect[line]; i++)
        if (jobMat[line*size + i] >= 0) count++;
    return count;
}

// determine the position where to cut the line between the 2 threads
int ivm::cuttingPosition(const int line, const int division)
{
	int nbSubtrees  = endVect[line] - posVect[line];
	int expSubtrees = countExplorableSubtrees(line);

	if (expSubtrees > nbSubtrees) {
		std::cout << "Explorable subtrees > available subtrees" << std::endl;
		exit(-1);
	}

	// victim thread keeps (expSubtrees / division) subtrees plus the one it is
	// already exploring
	int keep = expSubtrees / division;

	// determine where the thief's interval should start
	// int pos = firstAvailableSubtree(line);
	int pos          = posVect[line] + 1;
	int keptSubtrees = 0;

	while (keptSubtrees < keep) {
		if (jobMat[line * size + pos] >= 0) keptSubtrees++;
		pos++;
	}

	if (pos <= posVect[line]) {
		std::cout << "cutting position (" << pos << ") <= current position (" <<
		posVect[line] << ")" << std::endl;
		exit(-1);
	}

	if (pos > endVect[line]) {
		std::cout << "cutting position (" << pos << ") > end (" << endVect[line] <<
		")" <<
		std::endl;
		exit(-1);
	}

	return pos;
}

bool ivm::intervalValid(){
    for (int i = 0; i < size; i++) {
        if ((posVect[i] < 0) || (posVect[i] >= size - i)) {
            std::cout << " incorrect position vector: pos[" << i << "]=" << posVect[i] << " size="<<size<<std::endl;
            exit(-1);
        }
        if ((endVect[i] < 0) || (endVect[i] >= size - i)) {
            std::cout << " incorrect end vector " << i << " " << endVect[i] << std::endl;
            std::cout << " pos " << i << " " << posVect[i] << std::endl;
            exit(-1);
        }
    }
}


void
ivm::getSchedule(int *sch)
{
    for (int i = 0; i < size; i++) {
        sch[i]=node->schedule[i];
    }
}



template void ivm::sortSiblingNodes<int>(std::vector<int> lb,std::vector<int> prio);
