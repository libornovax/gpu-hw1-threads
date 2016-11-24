//
// Libor Novak
// 11/21/2016
//

#ifndef GPUFILTER_H
#define GPUFILTER_H

#include "data.h"


namespace GPUFilter {

    DataArray filterArray (const DataArray &da);

    /**
     * @brief Initializes the CUDA environment
     */
    bool initialize ();

}


#endif // GPUFILTER_H
