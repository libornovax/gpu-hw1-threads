#ifndef SETTINGS_H
#define SETTINGS_H


// The length of the sequence to be sorted. MUST BE >= 2*THREADS_PER_BLOCK
//#define SEQUENCE_LENGTH 128
//#define SEQUENCE_LENGTH 512
#define SEQUENCE_LENGTH 2048
//#define SEQUENCE_LENGTH 1048576  // 1024*1024

// Threads that will run in each block (must be 2^n)
#define THREADS_PER_BLOCK 128


#endif // SETTINGS_H

