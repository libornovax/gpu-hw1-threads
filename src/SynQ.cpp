#ifndef SYNQ_CPP
#define SYNQ_CPP

#include "SynQ.h"

#include <cassert>
#include "utils.h"


template<typename T, int SIZE>
SynQ<T, SIZE>::SynQ (const T &shut_down_signal)
    : _input_forbidden(false),
      _shut_down_signal(shut_down_signal)
{
}


template<typename T, int SIZE>
SynQ<T, SIZE>::~SynQ ()
{
}


template<typename T, int SIZE>
void SynQ<T, SIZE>::push_back (const T &obj)
{
    std::unique_lock<std::mutex> lk(this->_mtx);

    if (this->_queue.size() >= SIZE)
    {
        // Queue full
        this->_cv_full.wait(lk, [this](){ return this->_queue.size() < SIZE; });
    }

#ifdef DEBUG_PRINT
    syncPrint("SynQ: <<-- (" + std::to_string(obj->getIdx()) + ")");
#endif
    assert(!this->_input_forbidden);
    this->_queue.push_back(obj);

    lk.unlock();
    this->_cv_empty.notify_one();
}


template<typename T, int SIZE>
T SynQ<T, SIZE>::pop_front ()
{
    std::unique_lock<std::mutex> lk(this->_mtx);

    if (this->_queue.empty())
    {
        // Queue empty
        this->_cv_empty.wait(lk);

        if (this->_queue.empty())
        {
            // This was a shut down signal because the queue is still empty - it had to come from
            // the shutDown() method
            return this->_shut_down_signal;
        }
    }

    auto obj = this->_queue.front();
    this->_queue.pop_front();

#ifdef DEBUG_PRINT
    syncPrint("SynQ: -->> (" + std::to_string(obj->getIdx()) + ")");
#endif

    lk.unlock();
    this->_cv_full.notify_all();

    return obj;
}


template<typename T, int SIZE>
bool SynQ<T, SIZE>::empty ()
{
    std::lock_guard<std::mutex> lg(this->_mtx);
    return this->_queue.empty();
}


template<typename T, int SIZE>
void SynQ<T, SIZE>::shutDown ()
{
    std::unique_lock<std::mutex> lk(this->_mtx);

    if (!this->_queue.empty())
    {
        // Queue empty
        this->_cv_full.wait(lk, [this](){ return this->_queue.empty(); });
    }

    this->_input_forbidden = true;
    lk.unlock();

    // Send shut down signal - queue is empty after notification (which otherwise does not happen)
    this->_cv_empty.notify_all();
}


#endif // SYNQ_CPP
