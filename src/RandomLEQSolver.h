#ifndef RANDOMLEQSOLVER_H
#define RANDOMLEQSOLVER_H

#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include "LEQSystem.h"
#include "GEMSolvingThreadPool.h"


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

    /**
     * @brief Runs the whole process of generating and solving the systems of linear equations
     */
    void solve ();


private:

    void _generateEquationSystems ();

    void _determineRank ();
    static void _runDetermineRank (std::shared_ptr<LEQSystem> &leq_system);

    void _saveResults ();
    static void _runSaving (std::shared_ptr<LEQSystem> &leq_system);


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    GEMSolvingThreadPool _thread_pool;

    // Stage 4 - save results
    // Input variable (queue of size 1)
    std::shared_ptr<LEQSystem> _save_in;
    // Input logic
    std::mutex _mtx_save_in;
    std::condition_variable _cv_save_in_full;
    std::condition_variable _cv_save_in_empty;

    // Shut down signal for the saving thread
    std::atomic<bool> _shut_down;

};

#endif // RANDOMLEQSOLVER_H
