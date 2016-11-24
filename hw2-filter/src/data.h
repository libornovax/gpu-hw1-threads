//
// Libor Novak
// 11/21/2016
//

#ifndef DATA_H
#define DATA_H


/**
 * @brief The Data struct
 * Data member of the array
 */
struct Data {
    int key;
    float data;
};


/**
 * @brief The DataArray struct
 * Handle for allocating and deallocating an array of Data objects
 */
struct DataArray {

    DataArray (int size)
    {
        this->size = size;
        this->array = new Data[size];
    }

    DataArray (const DataArray &other)
    {
        this->array = new Data[other.size];
        for (int i = 0; i < other.size; ++i) this->array[i] = other.array[i];
    }

    ~DataArray ()
    {
        delete [] this->array;
    }

    Data* array;
    int size;
};

#endif // DATA_H

