#include "RandomLEQSolver.h"

#include <iostream>
#include <cmath>
#include "settings.h"
#include "utils.h"


RandomLEQSolver::RandomLEQSolver ()
    : _shut_down(false),
      _save_in(nullptr)
{
}


RandomLEQSolver::~RandomLEQSolver ()
{

}


void RandomLEQSolver::solve ()
{
    // Equation systems generator
    std::thread t_generator(&RandomLEQSolver::_generateEquationSystems, this);
    // Rank computation
    std::thread t_rank(&RandomLEQSolver::_determineRank, this);
    // Saving of the results
    std::thread t_save(&RandomLEQSolver::_saveResults, this);


    // Now we wait for all the equations to be generated. After that is done we know that we need to shut
    // down the whole system
    t_generator.join();
    // Shut down the thread pool
    this->_thread_pool.shutDown();
    // Shut down of thread pool initiates the shut down of t_rank
    t_rank.join();
    // After the save thread finishes, the whole thing is done
    t_save.join();
}



// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void RandomLEQSolver::_generateEquationSystems ()
{
    syncPrint("-- GENERATOR starting");

    // Generate MATRIX_COUNT random linear equation systems
    for (int i = 0; i < MATRIX_COUNT; ++i)
    {
        this->_thread_pool.addLEQSystem(std::make_shared<LEQSystem>(i));
    }

    syncPrint("-- GENERATOR shutting down");
}


void RandomLEQSolver::_determineRank ()
{
    syncPrint("-- RANK starting");

    while (true)
    {
        auto ls = this->_thread_pool.getGEMSolvedLEQSystem();

        if (ls)
        {
            // Determine the rank of this system
            this->_runDetermineRank(ls);

            this->_save_in.push_back(ls);
        }
        else
        {
            // Null pointer means shut down
            this->_save_in.shutDown();
            this->_shut_down = true;
            break;
        }
    }

    syncPrint("-- RANK shutting down");
}


void RandomLEQSolver::_runDetermineRank (std::shared_ptr<LEQSystem> &leq_system)
{
    syncPrint("Computing rank (" + std::to_string(leq_system->getIdx()) + ")");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}


void RandomLEQSolver::_saveResults ()
{
    syncPrint("-- SAVE starting");

    while (!(this->_shut_down && this->_save_in.empty()))
    {
        auto ls = this->_save_in.pop_front();

        if (ls)
        {
            this->_runSaving(ls);
        }
        else
        {
            // Null pointer - Shut down signal received
            break;
        }
    }

    syncPrint("-- SAVE shutting down");
}


void RandomLEQSolver::_runSaving (std::shared_ptr<LEQSystem> &leq_system)
{
    syncPrint("Saving (" + std::to_string(leq_system->getIdx()) + ")");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
}



