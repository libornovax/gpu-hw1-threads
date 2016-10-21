#include "GEMSolvingThreadPool.h"

#include <cassert>
#include "settings.h"
#include "utils.h"


GEMSolvingThreadPool::GEMSolvingThreadPool ()
    : _num_running_workers(0),
      _buffer_in(nullptr),
      _buffer_out(nullptr)
{
    for (int i = 0; i < STAGE2_WORKERS_COUNT; ++i)
    {
        this->_worker_pool.emplace_back(&GEMSolvingThreadPool::_GEMWorker, this);
    }
}


GEMSolvingThreadPool::~GEMSolvingThreadPool ()
{
}


void GEMSolvingThreadPool::addLEQSystem (const std::shared_ptr<LEQSystem> &ls)
{
    this->_buffer_in.push_back(ls);
}


std::shared_ptr<LEQSystem> GEMSolvingThreadPool::getGEMSolvedLEQSystem ()
{
    return this->_buffer_out.pop_front();
}


void GEMSolvingThreadPool::shutDown ()
{
    // Notify the workers to shut down (all of them will receive the shut down signal once the buffer is
    // empty)
    this->_buffer_in.shutDown();

    // Wait for all workers to finish - VERY IMPORTANT (otherwise we could shut down the rest of
    // the pipeline before all workers have finished processing!)
    for (auto &w: this->_worker_pool) w.join();

    assert(this->_num_running_workers == 0);

    // Notify the output that we shut down - propagate the shut down
    // All threads processing this buffer will receive the shut down signal once the buffer is empty
    this->_buffer_out.shutDown();
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void GEMSolvingThreadPool::_GEMWorker ()
{
    int worker_id = this->_num_running_workers++;
    syncPrint("-- WORKER [" + std::to_string(worker_id) + "] starting");

    while (true)
    {
        // Load an available unsolved system from the input buffer
        auto ls = this->_buffer_in.pop_front();

        if (ls)
        {
            // Perform GEM on the system
            // ...
            syncPrint("TP: WORKER [" + std::to_string(worker_id) + "] processing (" + std::to_string(ls->getIdx()) + ")");
            std::this_thread::sleep_for(std::chrono::seconds(1));

            // Deposit a solved system to the output buffer
            this->_buffer_out.push_back(ls);
        }
        else
        {
            // Null pointer means shut down
            break;
        }
    }

    this->_num_running_workers--;
    syncPrint("-- WORKER [" + std::to_string(worker_id) + "] shutting down");
}


