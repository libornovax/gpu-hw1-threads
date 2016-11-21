//
// Libor Novak
// 10/21/2016
//

#ifndef SYNQ_H
#define SYNQ_H

#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>


template<typename T, int SIZE>
/**
 * @brief The SynQ class (Synchronized Queue)
 * Queue ment to be used in multithreading applications as a buffer to hand over data between threads
 */
class SynQ
{
public:

    SynQ (const T &shut_down_signal);
    ~SynQ ();

    /**
     * @brief Add a new object to the queue
     * @param obj Object to be added
     */
    void push_back (const T &obj);

    /**
     * @brief Remove the first object from the queue
     * @return Object
     */
    T pop_front ();

    /**
     * @brief Are there any objects in the queue
     * @return true if not
     */
    bool empty ();

    /**
     * @brief Makes all waiting pop_front() send the shut_down_signal
     */
    void shutDown ();


private:

    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    std::deque<T> _queue;

    // Queue access logic
    std::mutex _mtx;
    std::condition_variable _cv_full;
    std::condition_variable _cv_empty;

    std::atomic<bool> _input_forbidden;
    T _shut_down_signal;

};


#include "SynQ.cpp"


#endif // SYNQ_H
