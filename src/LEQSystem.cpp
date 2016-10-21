#include "LEQSystem.h"

#include <random>
#include "settings.h"


LEQSystem::LEQSystem (int idx)
    : _idx(idx),
      _b(MATRIX_DIM)
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


int LEQSystem::getIdx () const
{
    return this->_idx;
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void LEQSystem::_randomInit ()
{
    // Generate random values from [-1000,1000]
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<> dist(-1000, 1000);

    for (auto &row: this->_A)
    {
        for (auto &cell: row) cell = dist(mt);
    }

    for (auto &cell: this->_b) cell = dist(mt);
}
