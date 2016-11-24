#include "ArrayGenerator.h"

#include <random>


DataArray ArrayGenerator::generateRandomArray (int size)
{
    std::random_device rd;
    std::mt19937 mt(rd());

    DataArray da(size);


    // Key is from 0 to 100
    std::uniform_int_distribution<> key_dist(0, 100);

    for (size_t i = 0; i < size; ++i)
    {
        da.array[i].key = key_dist(mt);
        // Data will not be random for now - I want to see the order of the elements after the filtering
        da.array[i].data = float(i);
    }

    return da;
}

