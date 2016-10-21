#include "RandomLEQSolver.h"

#include <iostream>
#include <cmath>
#include "settings.h"
#include "utils.h"


RandomLEQSolver::RandomLEQSolver ()
    : _shut_down(false)
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

    t_generator.join();
    syncPrint("-----------------  SHUT DOWN INITIATED  -----------------");
    this->_thread_pool.shutDown();

    // After the save thread finishes, the whole thing is done
    t_save.join();
}



// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void RandomLEQSolver::_generateEquationSystems ()
{
    for (int i = 0; i < MATRIX_COUNT; ++i)
    {
        syncPrint("GENERATOR: Adding equation (" + std::to_string(i) + ")");
        this->_thread_pool.addLEQSystem(std::make_shared<LEQSystem>(i));
    }

    // Do something to shut down the whole thing...
//    this->_shut_down = true;
//    this->_cv_empty.notify_all();  // All waiting workers need to be woken up and shut down
}


void RandomLEQSolver::_determineRank ()
{
    while (true)
    {
        auto ls = this->_thread_pool.getGEMSolvedLEQSystem();
        this->_runDetermineRank(ls);


        // Pass to saving thread
        std::unique_lock<std::mutex> lk2(this->_mtx_save_in);
        if (this->_save_in)
        {
            // There are data now
            this->_cv_save_in_empty.wait(lk2, [this](){ return bool(!this->_save_in); });
        }
        this->_save_in = ls;

        lk2.unlock();
        this->_cv_save_in.notify_all();
    }
}


void RandomLEQSolver::_runDetermineRank (std::shared_ptr<LEQSystem> &leq_system)
{
    syncPrint("Computing rank (" + std::to_string(leq_system->getIdx()) + ")");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}


void RandomLEQSolver::_saveResults ()
{
    while (true)
    {
        std::unique_lock<std::mutex> lk3(this->_mtx_save_in);

        // Check if we should not close the whole program
        if (!this->_save_in                             // Nothing is on the save input
                && this->_shut_down)                    // Shut down was initiated
        {
//            syncPrint("---- TASKS PROCESSED " + std::to_string(this->_tasks_processed) + " ----");
            break;
        }


        if (!this->_save_in)
        {
            // No data on input, wait
            this->_cv_save_in.wait(lk3, [this](){ return bool(this->_save_in); });
        }

        // Save the data
        this->_runSaving(this->_save_in);
        this->_save_in = nullptr;

        lk3.unlock();
        this->_cv_save_in_empty.notify_all();
    }
}


void RandomLEQSolver::_runSaving (std::shared_ptr<LEQSystem> &leq_system)
{
    syncPrint("Saving (" + std::to_string(leq_system->getIdx()) + ")");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
}



