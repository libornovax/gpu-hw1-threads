//
// Libor Novak
// 10/20/2016
//

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
    SynQ<std::shared_ptr<LEQSystem>, 1> _save_in;

};

#endif // RANDOMLEQSOLVER_H
