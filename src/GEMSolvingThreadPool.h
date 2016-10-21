#ifndef GEMSOLVINGTHREADPOOL_H
#define GEMSOLVINGTHREADPOOL_H

#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include "LEQSystem.h"


class GEMSolvingThreadPool
{
public:
    GEMSolvingThreadPool ();
    ~GEMSolvingThreadPool ();

    /**
     * @brief Adds a new LEQSystem to the processing queue
     */
    void addLEQSystem (const std::shared_ptr<LEQSystem> &ls);

    /**
     * @brief Returns an LEQSystem, which was ran through Gaussian elimination (GEM)
     */
    std::shared_ptr<LEQSystem> getGEMSolvedLEQSystem ();


private:

    /**
     * @brief A worker thread, which processes the input buffer and fills the output buffer
     */
    void _GEMWorker ();

    /**
     * @brief Loads an available unsolved system from the input buffer
     */
    std::shared_ptr<LEQSystem> _getLEQSystem ();

    /**
     * @brief Deposits a solved system to the output buffer
     */
    void _depositProcessedLEQSystem (const std::shared_ptr<LEQSystem> &ls);


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // Input buffer of size STAGE2_BUFFER_SIZE
    std::deque<std::shared_ptr<LEQSystem>> _buffer_in;
    // Output buffer of size STAGE3_BUFFER_SIZE (should be 1 as required by the assignment)
    std::deque<std::shared_ptr<LEQSystem>> _buffer_out;

    std::mutex _mtx_buffer_in;
    std::mutex _mtx_buffer_out;

    std::condition_variable _cv_buffer_in_full;
    std::condition_variable _cv_buffer_in_empty;
    std::condition_variable _cv_buffer_out_full;
    std::condition_variable _cv_buffer_out_empty;

    std::vector<std::thread> _worker_pool;
};

#endif // GEMSOLVINGTHREADPOOL_H
