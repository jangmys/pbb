#include "../../common/include/pbab.h"
#include "ivm.h"

ivm::ivm(int _size) : size(_size){
    jobMat = (int*)calloc(size*size,sizeof(int));
    posVect = (int*)calloc(size,sizeof(int));
    endVect = (int*)calloc(size,sizeof(int));
    dirVect = (int*)calloc(size,sizeof(int));

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

void ivm::clearInterval()
{
    memset(jobMat, 0, size*size*sizeof(int));
    memset(posVect, 0, size*sizeof(int));
    memset(endVect, 0, size*sizeof(int));
    posVect[0]=size;
}

void ivm::getInterval(int* pos, int* end)
{
    // displayVector(posVect);
    // displayVector(endVect);
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
        // printf("\n\n\n\n+++++++++++++++++++++++++++++++++++++++++\n\n\n\n");
        posVect[line]++;
    }
}

//"branch"
void ivm::goDown()
{
    // if(line == size-1){
    //     std::cout << "After last line " << line << std::endl;
    //     exit(-1);
    // }
    // int pos = posVect[line];
    // if(pos<0 || pos>=size)
    // {
    //     std::cout << "Position " << pos << " outside matrix" << std::endl;
    //     exit(-1);
    // }
    // if(jobMat[line*size + pos] < 0)
    // {
    //     std::cout  << "Already explored position " << pos << " for line " << line << std::endl;
    //     exit(-1);
    // }
    line++;
    generateLine(line, true);

    // for (int i = 0; i < size-line; i++) {
    //     int job = jobMat[line*size+i];
    //     if(job<0 || job>=size){
    //         printf("MORE WEIRDNESS:invalid job %d (line %d)\n",job,line);
    //         displayVector(posVect);
    //         displayVector(endVect);
    //         displayMatrix();//Vector(jm);
    //         // break;
    //         exit(-1);
    //     }
    // }
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
	// int nonzero=0;
	// for(int i=0;i<size;++i){
	// 	if(posVect[i]>0)nonzero+=(size-i);
	// }
	// if(nonzero>200)return true;

	// int disc=0;
	// for(int i=0;i<size;++i){
	// 	disc += posVect[i];
	// }
	// if(disc>50)return true;

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
