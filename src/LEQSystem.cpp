#include "LEQSystem.h"

#include <random>
#include <iomanip>
#include <sstream>
#include "settings.h"


LEQSystem::LEQSystem (int idx)
    : _idx(idx),
      _b(MATRIX_DIM),
      _x(MATRIX_DIM)
{
    this->_A.resize(MATRIX_DIM);
    for (auto &row: this->_A)
    {
        row.resize(MATRIX_DIM);
    }

    this->_randomInit();
}


std::vector<std::vector<double>>& LEQSystem::getA ()
{
    return this->_A;
}


std::vector<double>& LEQSystem::getb ()
{
    return this->_b;
}


std::vector<double>& LEQSystem::getx ()
{
    return this->_x;
}


int LEQSystem::getIdx () const
{
    return this->_idx;
}


std::string LEQSystem::print () const
{
    std::stringstream out; out << std::fixed << std::showpoint;
    for (int i = 0; i < MATRIX_DIM; ++i)
    {
        for (int j = 0; j < MATRIX_DIM; ++j)
        {
            out << std::setw(9) << std::setprecision(2) << this->_A[i][j];
        }
        out << " | " << std::setw(9) << std::setprecision(2) << this->_b[i] << std::endl;
    }

    return out.str();
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void LEQSystem::_randomInit ()
{
    // Generate random values from [-1000,1000]
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<> dist(-100, 100);

    for (auto &row: this->_A)
    {
        for (auto &cell: row) cell = dist(mt);
    }

    for (auto &cell: this->_b) cell = dist(mt);
}
