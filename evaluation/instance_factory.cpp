#include <memory>
#include "instance_factory.h"

namespace pbb_instance
{
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
