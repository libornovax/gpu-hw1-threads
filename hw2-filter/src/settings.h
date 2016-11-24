#ifndef SETTINGS_H
#define SETTINGS_H


// Size of the array that is being filtered
#define ARRAY_SIZE 262144
//#define ARRAY_SIZE 100000     // 0.8 MB
//#define ARRAY_SIZE 1000000    // 8 MB
//#define ARRAY_SIZE 10000000   // 80 MB
//#define ARRAY_SIZE 100000000  // 800 MB
//#define ARRAY_SIZE 200000000  // 1.6 GB

// Threads that will run in each block
#define THREADS_PER_BLOCK 32

// Interval that filters the array indices
#define FILTER_MIN 30
#define FILTER_MAX 60


#endif // SETTINGS_H

