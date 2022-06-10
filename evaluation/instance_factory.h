#ifndef INSTANCE_FACTORY_H
#define INSTANCE_FACTORY_H

#include <memory>

#include "instance_abstract.h"

#include "flowshop/include/instance_taillard.h"
#include "flowshop/include/instance_vrf.h"
#include "flowshop/include/instance_random.h"
#include "flowshop/include/instance_filename.h"
#include "dummy/include/instance_dummy.h"


class InstanceFactoryBase
{
public:
    virtual std::unique_ptr<instance_abstract>
        make_instance(char problem[],char inst_name[]) = 0;
};

class InstanceFactory : public InstanceFactoryBase
{
public:
    std::unique_ptr<instance_abstract>
        make_instance(char problem[],char inst_name[]) override
    {
        switch(problem[0])//DIFFERENT PROBLEMS...
        {
            case 'f': //FLOWSHOP
            {
                switch (inst_name[0]) {//DIFFERENT INSTANCES...
                    case 't': {
                        return std::make_unique<instance_taillard>(inst_name);
                            // instance = new instance_taillard(inst_name);
                        break;
                    }
                    case 'V': {
                        return std::make_unique<instance_vrf>(inst_name);
                        // instance = new instance_vrf(inst_name);
                        break;
                    }
                    case 'r': {
                        return std::make_unique<instance_random>(inst_name);
                        // instance = new instance_random(inst_name);
                        break;
                    }
                    case '.': {
                        return std::make_unique<instance_filename>(inst_name);
                        // instance = new instance_filename(inst_name);
                    }
                }
            }
            case 'd': //DUMMY
            {
                return std::make_unique<instance_dummy>(inst_name);
                // instance = new instance_dummy(inst_name);
            }
        }
    }
};

#endif
