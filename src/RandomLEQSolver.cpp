#include "RandomLEQSolver.h"

#include <iostream>
#include <cmath>
#include <iomanip>
#include "settings.h"
#include "utils.h"


RandomLEQSolver::RandomLEQSolver ()
    : _save_in(nullptr)
{
    this->_out_file.open("output.txt");
}


RandomLEQSolver::~RandomLEQSolver ()
{
    this->_out_file.close();
}


void RandomLEQSolver::solve ()
{
    // Equation systems generator
    std::thread t_generator(&RandomLEQSolver::_generateEquationSystems, this);
    // Rank computation
    std::thread t_rank(&RandomLEQSolver::_solve, this);
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


void RandomLEQSolver::_solve ()
{
    syncPrint("-- SOLVE starting");

    while (true)
    {
        auto ls = this->_thread_pool.getGEMSolvedLEQSystem();

        if (ls)
        {
            // Determine the rank of this system
            this->_runSolving(ls);

            this->_save_in.push_back(ls);
        }
        else
        {
            // Null pointer means shut down
            this->_save_in.shutDown();
            break;
        }
    }

    syncPrint("-- SOLVE shutting down");
}


void RandomLEQSolver::_runSolving (std::shared_ptr<LEQSystem> &leq_system)
{
    syncPrint("Solving (" + std::to_string(leq_system->getIdx()) + ")");

    // We skip rank computation because this exercise is about thread synchronization so we do not care
    // about a precise solution to the equation solving problem...

    auto &A = leq_system->getA();
    auto &b = leq_system->getb();
    auto &x = leq_system->getx(); // This is the solution

    for (int i = A.size()-1; i >= 0; i--)
    {
        double sum = 0;
        for (size_t j = i+1; j < A.size(); j++)
        {
            sum = sum + A[i][j]*x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }
}


void RandomLEQSolver::_saveResults ()
{
    syncPrint("-- SAVE starting");

    while (true)
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

    // We do not need any thread synchronization here because we have only one output thread
    this->_out_file << "==============================================   " << std::setw(4) << leq_system->getIdx() << "   ==============================================" << std::endl;
    this->_out_file << leq_system->print();
    this->_out_file << "------------------------------------------------------------------------------------------------------" << std::endl;
    this->_out_file << "Solution: ";

    auto x = leq_system->getx();
    for (int i = 0; i < x.size(); ++i)
    {
        if (i > 0) this->_out_file << "          ";
        this->_out_file << " x" << i << " = " << x[i] << std::endl;
    }
    this->_out_file << std::endl << std::endl;
}


