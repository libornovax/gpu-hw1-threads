#include "CPUFilter.h"

#include <cassert>
#include "settings.h"


namespace CPUFilter {


DataArray filterArray (const DataArray &da)
{
    // Two pass filtering - first we have to determine the number of elements in the output array, then copy
    // them to the output array

    int output_size = 0;
    for (int i = 0; i < da.size; ++i) output_size += (filter(da.array[i].key)) ? 1 : 0;

    DataArray da_out(output_size);
    int y = 0;
    for (int i = 0; i < da.size; ++i)
    {
        if (filter(da.array[i].key))
        {
            da_out.array[y].key = da.array[i].key;
            da_out.array[y].data = da.array[i].data;
            y++;
        }
    }

    assert(y == output_size);

    return da_out;
}


bool filter (int key)
{
    return key >= FILTER_MIN && key <= FILTER_MAX;
}


}
