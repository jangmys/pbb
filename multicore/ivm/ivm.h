#ifndef IVM_H
#define IVM_H

#include <subproblem.h>

class ivm{
private:
    int size;
    int line = 0;

    std::vector<int> posVect;
    std::vector<int> endVect;
    std::vector<int> dirVect;
    std::vector<int> jobMat;

    subproblem node;
public:
    explicit ivm(int _size);

    //GETTERS / SETTERS =========================
    subproblem& getNode();
    // void getSchedule(int *sch);
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
    //direction vector
    int getDirection(const int _d) const;
    void setDirection(int depth, int dir);
    //matrix
    void setRow(int k, const int *row);
    int *getRowPtr(int i);
    int getCell(int i,int j) const;

    inline int getCurrentCell() const
    {
        return jobMat[line*size+posVect[line]];
    };

    void eliminateCurrent();

    void clearInterval();
    void getInterval(int*,int*);

    //Tree operations
    void goDown(); //branch node
    void goRight(); //next sibling
    void goUp(); //backtrack

    //Exploration state
    bool isLastLine() const; //reached leaf node
    bool lineEndState() const; //all siblings explored
    bool pruningCellState() const; //is pruned
    bool beforeEnd() const; //interval not empty

    void generateLine(const int line, const bool explore);

    //Display methods
    void displayVector(int* ptr) const;
    void displayMatrix() const;
    void printRow(const int r) const;

    //Work partition
    int countExplorableSubtrees(const int line);
    int cuttingPosition(const int line, const int division);

    void decodeIVM();

    template<typename T>
    void sortSiblingNodes(std::vector<T> lb,std::vector<T> prio);

    bool intervalValid() const;
};


#endif
