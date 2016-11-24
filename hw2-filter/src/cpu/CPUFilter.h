//
// Libor Novak
// 11/21/2016
//

#ifndef CPUFILTER_H
#define CPUFILTER_H

#include "data.h"


namespace CPUFilter {

    DataArray filterArray (const DataArray &da);

    /**
     * @brief Returns true if the key is in the interval, false if it should be thrown away
     * @param key
     * @return
     */
    bool filter (int key);

}


#endif // CPUFILTER_H
