#ifndef GPUSORT_H
#define GPUSORT_H

#include <vector>


namespace GPUSort {

    void sortSequence (std::vector<float> &seq);

    /**
     * @brief Initializes the CUDA environment
     */
    bool initialize ();

}


#endif // GPUSORT_H
