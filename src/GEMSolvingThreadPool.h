//
// Libor Novak
// 10/21/2016
//

#ifndef GEMSOLVINGTHREADPOOL_H
#define GEMSOLVINGTHREADPOOL_H

#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include "LEQSystem.h"
#include "SynQ.h"
#include "settings.h"


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

    /**
     * @brief Waits for the queue to be processed and shuts down the threads (synchronously waits)
     */
    void shutDown();


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
    SynQ<std::shared_ptr<LEQSystem>, STAGE2_BUFFER_SIZE> _buffer_in;
    // Output buffer of size STAGE3_BUFFER_SIZE (should be 1 as required by the assignment)
    SynQ<std::shared_ptr<LEQSystem>, STAGE3_BUFFER_SIZE> _buffer_out;
    // Pool of workers
    std::vector<std::thread> _worker_pool;

    std::atomic<int> _num_running_workers;
    std::atomic<bool> _shut_down;

};

#endif // GEMSOLVINGTHREADPOOL_H
