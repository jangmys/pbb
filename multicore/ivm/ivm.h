#ifndef IVM_H
#define IVM_H

class ivm{
private:
    int size;

public:
    int line = 0;
    int* jobMat;
    int* posVect;
    int* endVect;
    int* dirVect;

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
};

#endif
