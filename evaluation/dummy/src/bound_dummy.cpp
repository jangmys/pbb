#include <bound_dummy.h>

void bound_dummy::init(instance_abstract& _instance)
{
    (_instance.data)->seekg(0);
    (_instance.data)->clear();

    size=8;
    *(_instance.data) >> size;
}
