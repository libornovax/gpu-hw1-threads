#include "utils.h"

#include <mutex>
#include <iostream>


void syncPrint (const std::string &txt)
{
    static std::mutex mtx;
    std::lock_guard<std::mutex> lk(mtx);
    std::cout << txt << std::endl;
}
