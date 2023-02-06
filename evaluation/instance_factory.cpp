#include <memory>
#include "instance_factory.h"

namespace pbb_instance
{
    // std::unique_ptr<instance_abstract>
    //     make_instance(char problem[],char inst_name[])
    // {
    //     switch(problem[0])//DIFFERENT PROBLEMS...
    //     {
    //         case 'f': //FLOWSHOP
    //         {
    //             switch (inst_name[0]) {//DIFFERENT INSTANCES...
    //                 case 't': {
    //                     return std::make_unique<instance_taillard>(inst_name);
    //                     break;
    //                 }
    //                 case 'V': {
    //                     return std::make_unique<instance_vrf>(inst_name);
    //                     break;
    //                 }
    //                 case 'r': {
    //                     return std::make_unique<instance_random>(inst_name);
    //                     break;
    //                 }
    //                 case '.': {
    //                     return std::make_unique<instance_filename>(inst_name);
    //                 }
    //             }
    //         }
    //         case 'd': //DUMMY
    //         {
    //             return std::make_unique<instance_dummy>(inst_name);
    //         }
    //     }
    //     return nullptr;
    // }

    instance_abstract
        make_inst(char problem[],char inst_name[])
    {
        switch(problem[0])//DIFFERENT PROBLEMS...
        {
            case 'f': //FLOWSHOP
            {
                switch (inst_name[0]) {//DIFFERENT INSTANCES...
                    case 't': {
                        return instance_taillard(inst_name);
                        break;
                    }
                    case 'V': {
                        return instance_vrf(inst_name);
                        break;
                    }
                    case 'r': {
                        return instance_random(inst_name);
                        break;
                    }
                    case '.': {
                        return instance_filename(inst_name);
                    }
                }
            }
            case 'd': //DUMMY
            {
                return instance_dummy(inst_name);
            }
        }
        return instance_dummy("8");
    }
}
