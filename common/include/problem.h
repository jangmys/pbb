class Problem
{
public:
    Problem(){};

    virtual void set_initial_solution() = 0;

private:
    instance_abstract inst;
};
