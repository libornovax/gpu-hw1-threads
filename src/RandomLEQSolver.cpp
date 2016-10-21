#include "RandomLEQSolver.h"

#include <iostream>
#include <cmath>
#include "settings.h"


namespace {

    /**
     * @brief Synchronized print
     */
    void syncPrint (const std::string &txt)
    {
        static std::mutex mtx;
        std::lock_guard<std::mutex> lk(mtx);
        std::cout << txt << std::endl;
    }

}


RandomLEQSolver::RandomLEQSolver ()
    : _shut_down(false),
      _workers_running(0),
      _tasks_processed(0)
{
}


RandomLEQSolver::~RandomLEQSolver ()
{

}


void RandomLEQSolver::solve ()
{
    // Equation systems generator
    std::thread t_generator(&RandomLEQSolver::_generateEquationSystems, this);

    // Create workers
    std::vector<std::thread> workers;
    for (int i = 0; i < STAGE2_WORKERS_COUNT; ++i)
    {
        this->_workers_running++;
        workers.push_back(std::thread(&RandomLEQSolver::_GEMWorker, this));
    }

    // Rank computation
    std::thread t_rank(&RandomLEQSolver::_determineRank, this);
    // Saving of the results
    std::thread t_save(&RandomLEQSolver::_saveResults, this);
    // After the save thread finishes, the whole thing is done
    t_save.join();
}



// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void RandomLEQSolver::_generateEquationSystems ()
{
    for (int i = 0; i < MATRIX_COUNT; ++i)
    {
        std::unique_lock<std::mutex> lk(this->_mtx_generator);

        if (this->_leq_system_queue.size() >= STAGE2_BUFFER_SIZE)
        {
            // Buffer full
            this->_cv_full.wait(lk, [this](){ return this->_leq_system_queue.size() < STAGE2_BUFFER_SIZE; });
        }

        syncPrint("GENERATOR: Adding equation (" + std::to_string(i) + ")");
        this->_leq_system_queue.push_back(std::make_shared<LEQSystem>(i));

        lk.unlock();
        this->_cv_empty.notify_all();
    }

    // Do something to shut down the whole thing...
    this->_shut_down = true;
    this->_cv_empty.notify_all();  // All waiting workers need to be woken up and shut down
}


void RandomLEQSolver::_GEMWorker ()
{
    // Worker identifier
    static int idx_count = 0;
    int idx = idx_count++;

    syncPrint("GEMWorker [" + std::to_string(idx) + "]: starting");

    while (true)
    {
        std::unique_lock<std::mutex> lk(this->_mtx_generator);

        if (this->_leq_system_queue.size() == 0)
        {
            if (this->_shut_down) break;

            // Buffer empty - nothing to process
            this->_cv_empty.wait(lk, [this](){ return this->_leq_system_queue.size() > 0 || this->_shut_down;});

            if (this->_shut_down && this->_leq_system_queue.size() == 0) break;
        }

        // Remove the last generated system from the buffer
        auto leq_system = this->_leq_system_queue[this->_leq_system_queue.size()-1];
        this->_leq_system_queue.pop_back();

        lk.unlock();
        this->_cv_full.notify_all();


        // -- Now run the GEM -- //
        syncPrint("GEMWorker [" + std::to_string(idx) + "]: processing (" + std::to_string(leq_system->getIdx()) + ")");
        this->_runGEM(leq_system);
        this->_tasks_processed++;

        // Pass the LEQ system to the rank determining thread
        std::unique_lock<std::mutex> lk2(this->_mtx_rank_in);
        if (this->_rank_in)
        {
            // There are data now - we need to wait
            this->_cv_rank_in_empty.wait(lk2, [this](){ return bool(!this->_rank_in); });
        }
        this->_rank_in = leq_system;
        lk2.unlock();
        this->_cv_rank_in.notify_all();  // This could be notify_one(), but it does not matter
    }

    syncPrint("GEMWorker [" + std::to_string(idx) + "]: killing");
    this->_workers_running--;
}


void RandomLEQSolver::_runGEM (std::shared_ptr<LEQSystem> &leq_system)
{
    // Here we carry out the GEM computation...
    std::this_thread::sleep_for(std::chrono::seconds(1));
}


void RandomLEQSolver::_determineRank ()
{
    while (true)
    {
        std::unique_lock<std::mutex> lk(this->_mtx_rank_in);

        if (!this->_rank_in)
        {
            if (this->_shut_down && this->_workers_running == 0) break;

            // No data on input, wait
            this->_cv_rank_in.wait(lk, [this](){ return bool(this->_rank_in) || this->_shut_down; });

            if (this->_shut_down && this->_workers_running == 0 && !this->_rank_in) break;
            if (!this->_rank_in) continue;  // Shut down was called, but there are still workers running
        }


        // Compute the rank
        this->_runDetermineRank(this->_rank_in);

        // Pass to saving thread
        std::unique_lock<std::mutex> lk2(this->_mtx_save_in);
        if (this->_save_in)
        {
            // There are data now
            this->_cv_save_in_empty.wait(lk2, [this](){ return bool(!this->_save_in); });
        }
        this->_save_in = this->_rank_in;
        this->_rank_in = nullptr;  // Set the input to empty

        lk.unlock();
        this->_cv_rank_in_empty.notify_all();

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
        // If we use a different order of the locks we will have a deadlock!
        std::unique_lock<std::mutex> lk1(this->_mtx_rank_in);
        std::unique_lock<std::mutex> lk2(this->_mtx_generator);

        std::unique_lock<std::mutex> lk3(this->_mtx_save_in);

        // Check if we should not close the whole program
        if (!this->_save_in                             // Nothing is on the save input
                && !this->_rank_in                      // Nothing is on the rank input
                && this->_leq_system_queue.size() == 0  // There are no more equations to process
                && this->_workers_running == 0          // All workers have finished processing
                && this->_shut_down)                    // Shut down was initiated (redundant)
        {
            syncPrint("---- TASKS PROCESSED " + std::to_string(this->_tasks_processed) + " ----");
            break;
        }

        lk1.unlock();
        lk2.unlock();


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



