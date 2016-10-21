#include "GEMSolvingThreadPool.h"

#include "settings.h"
#include "utils.h"


GEMSolvingThreadPool::GEMSolvingThreadPool ()
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
    std::unique_lock<std::mutex> lk(this->_mtx_buffer_in);

    if (this->_buffer_in.size() >= STAGE2_BUFFER_SIZE)
    {
        // Input buffer full
        this->_cv_buffer_in_full.wait(lk, [this](){ return this->_buffer_in.size() < STAGE2_BUFFER_SIZE; });
    }

    syncPrint("TP: INPUT <<-- (" + std::to_string(ls->getIdx()) + ")");
    this->_buffer_in.push_back(ls);

    lk.unlock();
    this->_cv_buffer_in_empty.notify_all();
}


std::shared_ptr<LEQSystem> GEMSolvingThreadPool::getGEMSolvedLEQSystem ()
{
    std::unique_lock<std::mutex> lk(this->_mtx_buffer_out);

    if (this->_buffer_out.size() == 0)
    {
        // Output buffer empty
        this->_cv_buffer_out_empty.wait(lk, [this](){ return this->_buffer_out.size() > 0; });
    }

    auto ls = this->_buffer_out.front();
    this->_buffer_out.pop_front();
    syncPrint("TP: OUTPUT -->> (" + std::to_string(ls->getIdx()) + ")");

    lk.unlock();
    this->_cv_buffer_out_full.notify_all();

    return ls;
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void GEMSolvingThreadPool::_GEMWorker ()
{
    static int worker_id_count = 0;
    int worker_id = worker_id_count++;

    while (true)
    {
        auto ls = this->_getLEQSystem();

        // Perform GEM on the system
        // ...
        std::this_thread::sleep_for(std::chrono::seconds(1));

        this->_depositProcessedLEQSystem(ls);
    }
}


std::shared_ptr<LEQSystem> GEMSolvingThreadPool::_getLEQSystem ()
{
    std::unique_lock<std::mutex> lk(this->_mtx_buffer_in);

    if (this->_buffer_in.size() == 0)
    {
        // Input buffer empty
        this->_cv_buffer_in_empty.wait(lk, [this](){ return this->_buffer_in.size() > 0; });
    }

    auto ls = this->_buffer_in.front();
    this->_buffer_in.pop_front();
    syncPrint("TP: INPUT -->> (" + std::to_string(ls->getIdx()) + ")");

    lk.unlock();
    this->_cv_buffer_in_full.notify_all();

    return ls;
}


void GEMSolvingThreadPool::_depositProcessedLEQSystem (const std::shared_ptr<LEQSystem> &ls)
{
    std::unique_lock<std::mutex> lk(this->_mtx_buffer_out);

    if (this->_buffer_in.size() >= STAGE3_BUFFER_SIZE)
    {
        // Output buffer full
        this->_cv_buffer_out_full.wait(lk, [this](){ return this->_buffer_out.size() < STAGE3_BUFFER_SIZE; });
    }

    syncPrint("TP: OUTPUT <<-- (" + std::to_string(ls->getIdx()) + ")");
    this->_buffer_out.push_back(ls);

    lk.unlock();
    this->_cv_buffer_out_empty.notify_all();
}

