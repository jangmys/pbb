#ifndef INSTANCE_FACTORY_H
#define INSTANCE_FACTORY_H

#include <memory>

#include "instance_abstract.h"

#include "flowshop/include/instance_taillard.h"
#include "flowshop/include/instance_vrf.h"
#include "flowshop/include/instance_random.h"
#include "flowshop/include/instance_filename.h"
#include "dummy/include/instance_dummy.h"

namespace pbb_instance
{
    std::shared_ptr<instance_abstract> make_inst(std::string problem,std::string inst_name);
}

#endif
