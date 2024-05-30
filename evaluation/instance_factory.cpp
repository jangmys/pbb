#include <memory>
#include <iostream>
#include "instance_factory.h"

namespace pbb_instance
{
    std::shared_ptr<instance_abstract>
        make_inst(std::string problem, std::string inst_name)
    {
        switch(problem[0])//DIFFERENT PROBLEMS...
        {
            case 'f': //FLOWSHOP
            {
                switch (inst_name[0]) {//DIFFERENT INSTANCES...
                    case 't': {
                        return std::make_shared<instance_taillard>(inst_name);
                    }
                    case 'V': {
                        return std::make_shared<instance_vrf>(inst_name);
                    }
                    case 'r': {
                        return std::make_shared<instance_random>(inst_name);
                    }
                    case '.': {
                        return std::make_shared<instance_filename>(inst_name);
                    }
                }
            }
            case 'd': //DUMMY
            {
                return std::make_shared<instance_dummy>(inst_name);
            }
        }
        return std::make_shared<instance_dummy>("8");
    }
}
