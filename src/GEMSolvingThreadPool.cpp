#include "GEMSolvingThreadPool.h"

#include <cassert>
#include "settings.h"
#include "utils.h"


GEMSolvingThreadPool::GEMSolvingThreadPool ()
    : _num_running_workers(0),
      _shut_down(false),
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
    // We need to check if we should shut down (SynQ cannot test for running workers!)
    if (this->_shut_down && this->_num_running_workers == 0 && this->_buffer_out.empty())
    {
        return nullptr;
    }

    return this->_buffer_out.pop_front();
}


void GEMSolvingThreadPool::shutDown ()
{
    this->_shut_down = true;

    // Notify the waiting threads to shut down
    // They will shut down automatically by reading the this->_shut_down variable and having nothing in the
    // input queue or they will receive this signal if they are waiting
    this->_buffer_in.shutDown();

    // Wait for all workers to finish
    for (auto &w: this->_worker_pool) w.join();

    assert(this->_num_running_workers == 0);

    // Notify the output that we shut down - propagate the shut down
    // The thread that is processing the output will either detect shut down in the getGEMSolvedLEQSystem()
    // function or it will receive this signal if it is waiting
    this->_buffer_out.shutDown();
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void GEMSolvingThreadPool::_GEMWorker ()
{
    int worker_id = this->_num_running_workers++;
    syncPrint("-- WORKER [" + std::to_string(worker_id) + "] starting");

    while (!(this->_shut_down && this->_buffer_in.empty()))
    {
        auto ls = this->_getLEQSystem();

        if (ls)
        {
            // Perform GEM on the system
            // ...
            syncPrint("TP: WORKER [" + std::to_string(worker_id) + "] processing (" + std::to_string(ls->getIdx()) + ")");
            std::this_thread::sleep_for(std::chrono::seconds(1));

            this->_depositProcessedLEQSystem(ls);
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


std::shared_ptr<LEQSystem> GEMSolvingThreadPool::_getLEQSystem ()
{
    return this->_buffer_in.pop_front();
}


void GEMSolvingThreadPool::_depositProcessedLEQSystem (const std::shared_ptr<LEQSystem> &ls)
{
    this->_buffer_out.push_back(ls);
}

