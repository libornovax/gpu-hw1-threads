//
// Libor Novak
// 10/20/2016
//
// GPU class, HW1
// Multi-threaded solver of a system of linear equations.
//

#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>

#include "RandomLEQSolver.h"


int main (int argc, char* argv[])
{
    RandomLEQSolver s;
    s.solve();

    // Solver generates a file output.txt in the bin folder

    return EXIT_SUCCESS;
}
