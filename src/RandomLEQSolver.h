#ifndef RANDOMLEQSOLVER_H
#define RANDOMLEQSOLVER_H

#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include "LEQSystem.h"


/**
 * @brief The RandomLEQSolver class
 *  1. Randomly generate a set of equations Ax=b
 *  2. Gauss eliminate the matrices of the extended system (using a pool of workers)
 *  3. Determine matrix rank
 *  4. Write out the solution
 */
class RandomLEQSolver
{
public:

    RandomLEQSolver ();
    ~RandomLEQSolver ();

    void solve ();


private:

    void _generateEquationSystems ();

    void _GEMWorker ();
    static void _runGEM (std::shared_ptr<LEQSystem> &leq_system);

    void _determineRank ();
    static void _runDetermineRank (std::shared_ptr<LEQSystem> &leq_system);

    void _saveResults ();
    static void _runSaving (std::shared_ptr<LEQSystem> &leq_system);


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // Stage 1-2 - equation systems generation
    std::vector<std::shared_ptr<LEQSystem>> _leq_system_queue;
    std::mutex _mtx_generator;
    std::condition_variable _cv_full;
    std::condition_variable _cv_empty;

    // Stage 2 - GEM
    std::vector<std::thread> _workers;
    std::atomic<int> _workers_running;
    std::atomic<int> _tasks_processed;

    // Stage 3 - rank computation
    std::shared_ptr<LEQSystem> _rank_in;
    std::mutex _mtx_rank_in;
    std::condition_variable _cv_rank_in;
    std::condition_variable _cv_rank_in_empty;

    // Stage 4 - save results
    std::shared_ptr<LEQSystem> _save_in;
    std::mutex _mtx_save_in;
    std::condition_variable _cv_save_in;
    std::condition_variable _cv_save_in_empty;

    std::atomic<bool> _shut_down;

};

#endif // RANDOMLEQSOLVER_H
