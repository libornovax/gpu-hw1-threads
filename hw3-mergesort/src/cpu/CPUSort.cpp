#include "CPUSort.h"

#include <algorithm>


namespace CPUSort {

    void sortSequence (std::vector<float> &seq)
    {
        std::sort(seq.begin(), seq.end());
    }

}

