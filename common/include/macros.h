#ifndef MACROS_H
#define MACROS_H

#define WEAK (0)
#define STRONG (1)

#define WORK 1
#define BEST 2
#define END 3
#define NIL 4
#define SLEEP 5

#define _1MB 1048576
#define _2MB 2097152
#define _8MB 8388608
#define _128MB 134217728

#define MAX_MPZLEN 2000

//2*log(2;800!)/64 ~= 2*103 64bit limbs => 2*103*8+4 Bytes

//16384*2*log(2;50!)/8 ~= 1MB
//16384*2*log(2;100!)/8 ~= 2.1MB
//16384*2*log(2;200!)/8 ~= 5.1MB
//16384*2*log(2;400!)/8 ~= 11.82MB
//16384*2*log(2;600!)/8 ~= 19.1MB
//16384*2*log(2;600!)/8 ~= 26.9MB

// #define MAX_COMM_BUFFER 8388608
// #define MAX_COMM_BUFFER 16777216 //16 MB
#define MAX_COMM_BUFFER 33554432 //32 MB //
// #define MAX_COMM_BUFFER 134217728 //128MB


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
