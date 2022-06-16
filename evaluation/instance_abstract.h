#ifndef INSTANCE_ABSTRACT_H
#define INSTANCE_ABSTRACT_H

#include <iostream>
#include <sstream>
#include <pthread.h>


struct instance_abstract {
    instance_abstract()
    {
        // std::cout<<"instance base ctor\n";
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
        pthread_mutex_init(&mutex_instance_data, &attr);
    }

    int                 size; //problem size
    std::stringstream   *data; //the instance data

    pthread_mutex_t mutex_instance_data;
};

#endif
