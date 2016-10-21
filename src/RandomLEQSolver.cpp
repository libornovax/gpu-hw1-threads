#include "RandomLEQSolver.h"

#include <iostream>
#include "settings.h"


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
    std::thread t_generator(&RandomLEQSolver::_generateEquationSystems, this);

    // Create workers
    std::vector<std::thread> workers;
    for (int i = 0; i < STAGE2_WORKERS_COUNT; ++i)
    {
        this->_workers_running++;
        workers.push_back(std::thread(&RandomLEQSolver::_GEMWorker, this));
    }

    std::thread t_rank(&RandomLEQSolver::_determineRank, this);
    std::thread t_save(&RandomLEQSolver::_saveResults, this);
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

        std::cout << "GENERATOR: Adding equation (" << i << ")" << std::endl;
        this->_leq_system_queue.push_back(std::make_shared<LEQSystem>(i));

        lk.unlock();
        this->_cv_empty.notify_all();
    }

    // Do something to shut down the whole thing...
    this->_shut_down = true;
    this->_cv_empty.notify_all();
}


void RandomLEQSolver::_GEMWorker ()
{
    static int idx_count = 0;
    int idx = idx_count++;

    std::cout << "GEMWorker [" << idx << "]: starting" << std::endl;

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

        auto leq_system = this->_leq_system_queue[this->_leq_system_queue.size()-1];
        this->_leq_system_queue.pop_back();

        lk.unlock();
        this->_cv_full.notify_all();

        // Now run the GEM
        std::cout << "GEMWorker [" << idx << "]: processing (" << leq_system->getIdx() << ")"<< std::endl;
        this->_runGEM(leq_system);
        this->_tasks_processed++;


        // Pass the LEQ system to the rank determining thread
        std::unique_lock<std::mutex> lk2(this->_mtx_rank_in);
        if (this->_rank_in)
        {
            // There are data now
            this->_cv_rank_in_empty.wait(lk2, [this](){ return bool(!this->_rank_in); });
        }
        this->_rank_in = leq_system;
        lk2.unlock();
        this->_cv_rank_in.notify_all();
    }

    std::cout << "GEMWorker [" << idx << "]: killing" << std::endl;
    this->_workers_running--;
}


void RandomLEQSolver::_runGEM (std::shared_ptr<LEQSystem> &leq_system)
{
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
            if (!this->_rank_in) continue;
        }

        std::cout << "Rank (" << this->_rank_in->getIdx() << ")"<< std::endl;
        // Compute the rank
        // ...

        // Pass to saving thread
        std::unique_lock<std::mutex> lk2(this->_mtx_save_in);
        if (this->_save_in)
        {
            // There are data now
            this->_cv_save_in_empty.wait(lk2, [this](){ return bool(!this->_save_in); });
        }
        this->_save_in = this->_rank_in;
        this->_rank_in = nullptr;

        lk.unlock();
        this->_cv_rank_in_empty.notify_all();

        lk2.unlock();
        this->_cv_save_in.notify_all();
    }
}


void RandomLEQSolver::_saveResults ()
{
    while (true)
    {
        std::unique_lock<std::mutex> lk3(this->_mtx_rank_in);
        std::unique_lock<std::mutex> lk(this->_mtx_generator);
        std::unique_lock<std::mutex> lk2(this->_mtx_save_in);

        if (!this->_save_in && !this->_rank_in && this->_leq_system_queue.size() == 0 && this->_workers_running == 0 && this->_shut_down)
        {
            std::cout << "---- TASKS PROCESSED " << this->_tasks_processed << " ----" << std::endl;
            break;
        }

        lk.unlock();
        this->_cv_full.notify_all();
        lk3.unlock();
        this->_cv_rank_in_empty.notify_all();


        if (!this->_save_in)
        {
            // No data on input, wait
            this->_cv_save_in.wait(lk2, [this](){ return bool(this->_save_in); });
        }

        std::cout << "Saving (" << this->_save_in->getIdx() << ")"<< std::endl;
        // Save the data
        // ...
        this->_save_in = nullptr;


        lk2.unlock();
        this->_cv_save_in_empty.notify_all();
    }
}



