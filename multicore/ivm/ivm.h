#ifndef IVM_H
#define IVM_H

#include <subproblem.h>

class ivm{
private:
    int size;
    int line = 0;

    std::vector<int> posVect;
    // int* posVect;
    std::vector<int> endVect;
    // int* endVect;
    std::vector<int> dirVect;
    // int* dirVect;
    std::vector<int> jobMat;
    // int* jobMat;

    subproblem node;
public:
    subproblem& getNode();


    //operate on line
    int getDepth() const;
    void setDepth(int _line);
    void incrDepth();
    //operate on position vector
    void alignLeft();
    void setPosition(int* pos);
    void setPosition(int depth, int pos);
    int getPosition(const int _d) const;
    //operate on end vector
    void setEnd(int* end);
    void setEnd(int depth, int end);
    int getEnd(const int _d) const;

    int getDirection(const int _d) const;
    void setDirection(int depth, int dir);

    void setRow(int k, const int *row);
    int *getRowPtr(int i);
    int getCell(int i,int j) const;

    inline int getCurrentCell() const
    {
        return jobMat[line*size+posVect[line]];
    };


    explicit ivm(int _size);
    ~ivm();

    void eliminateCurrent();

    void clearInterval();
    void getInterval(int*,int*);
    void initEmpty();
    void initRoot();

    void goDown();
    void goRight();
    void goUp();

    bool isLineFinished() const;
    bool isLastLine() const;
    bool lineEndState() const;
    bool pruningCellState() const;
    void generateLine(const int line, const bool explore);
    bool beforeEnd() const;

    int getDir();

    void displayVector(int* ptr) const;
    void displayMatrix() const;
    void printRow(const int r) const;

    int countExplorableSubtrees(const int line);
    int cuttingPosition(const int line, const int division);

    void decodeIVM();

    template<typename T>
    void sortSiblingNodes(std::vector<T> lb,std::vector<T> prio);

    bool intervalValid();
    void getSchedule(int *sch);
};


#endif
