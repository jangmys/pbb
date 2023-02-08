#ifndef BOUND_FACTORY_H
#define BOUND_FACTORY_H

#include <memory>

#include "bound_abstract.h"

#include "flowshop/include/bound_fsp_strong.h"
#include "flowshop/include/bound_fsp_weak.h"

class BoundFactoryBase
{
public:
    BoundFactoryBase(){};

    virtual std::unique_ptr<bound_abstract<int>>
        make_bound(instance_abstract& inst, int bound_type) = 0;

protected:
    int bound_mode;
};

class BoundFactory : public BoundFactoryBase
{
public:
    BoundFactory(bool _early_stop = false, int _machine_pairs = 0) : BoundFactoryBase(), early_stop(_early_stop), machine_pairs(_machine_pairs){};

    // std::unique_ptr<bound_abstract<int>>
    //     make_bound(std::unique_ptr<instance_abstract>& inst, int bound_type) override
    std::unique_ptr<bound_abstract<int>>
        make_bound(instance_abstract& inst, int bound_type) override
    {
        switch (bound_type) {
            case 0:
            {
                std::unique_ptr<bound_fsp_weak> bd = std::make_unique<bound_fsp_weak>();
                bd->init(inst);
                return bd;
            }
            case 1:
            {
                std::unique_ptr<bound_fsp_strong> bd = std::make_unique<bound_fsp_strong>();
                bd->init(inst);
                bd->earlyExit=early_stop;
                bd->machinePairs=machine_pairs;
                return bd;
            }
        }
        return nullptr;
    }

private:
    bool early_stop;
    int machine_pairs;

};

class DummyBoundFactory : public BoundFactoryBase
{
public:
    DummyBoundFactory() : BoundFactoryBase(){};

    std::unique_ptr<bound_abstract<int>>
        make_bound(instance_abstract& inst, int bound_type) override
    {
        std::unique_ptr<bound_dummy> bd = std::make_unique<bound_dummy>();
        bd->init(inst);
        return bd;
    }
};


#endif
