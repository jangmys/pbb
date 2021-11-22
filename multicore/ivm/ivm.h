#ifndef IVM_H
#define IVM_H

#include <subproblem.h>

class ivm{
private:
    int size;

public:
    int line = 0;
    int* jobMat;
    int* posVect;
    int* endVect;
    int* dirVect;

    std::unique_ptr<subproblem> node;

    ivm(int _size);
    ~ivm();

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

    int countExplorableSubtrees(const int line);
    int cuttingPosition(const int line, const int division);

    void decodeIVM();

    template<typename T>
    void sortSiblingNodes(std::vector<T> lb,std::vector<T> prio);

    bool intervalValid();
    void getSchedule(int *sch);
};


#endif
