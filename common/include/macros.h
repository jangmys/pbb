#ifndef MACROS_H
#define MACROS_H

//================================================
#define MUTEXDEBUG
#undef MUTEXDEBUG

#ifdef MUTEXDEBUG
#define pthread_mutex_lock_check(mutex)		\
__extension__({						\
        int __ret = pthread_mutex_lock (mutex);	\
	if (__ret != 0)				\
		printf ("pthread_mutex_lock_check in %s line %u: error %d - %s\n", \
			 __FILE__, __LINE__, __ret, strerror (__ret)); \
	__ret;				\
})
#else
#define pthread_mutex_lock_check pthread_mutex_lock
#endif
#endif
