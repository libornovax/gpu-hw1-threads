//
// Libor Novak
// 10/20/2016
//
// GPU class, HW1
// Multi-threaded solver of a system of linear equations.
//
//  1. Randomly generate a set of equations Ax=b
//  2. Gauss eliminate the matrices of the extended system (using a pool of workers)
//  3. Determine matrix rank
//  4. Write out the solution
//

#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>

std::vector<std::string> buffer;
#define BUFF_SIZE 5


std::mutex mtx;
std::condition_variable cv1;
std::condition_variable cv2;


void producer ()
{
    static int my_num_counter = 0;
    int my_num = my_num_counter++;
    int data = 1;

    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::unique_lock<std::mutex> lk(mtx);

        if (buffer.size() == BUFF_SIZE)
        {
            // Buffer full
            cv1.wait(lk, [](){ return buffer.size() < BUFF_SIZE; });
        }

        std::cout << "[" << my_num << "] " << data << std::endl;
        buffer.push_back("[" + std::to_string(my_num) + "] " + std::to_string(data++));

        lk.unlock();
        cv2.notify_all();
    }
}


void consumer ()
{
    static int my_num_counter = 0;
    int my_num = my_num_counter++;

    while (true)
    {
//        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::unique_lock<std::mutex> lk(mtx);

        if (buffer.size() == 0)
        {
            // Buffer empty - nothing to read
            cv2.wait(lk, [](){ return buffer.size() > 0; });
        }

        std::cout << "(" << my_num << ") data: " << buffer[buffer.size()-1] << std::endl;
        buffer.pop_back();

        lk.unlock();
        cv1.notify_all();
    }
}


int main (int argc, char* argv[])
{
    std::vector<std::thread> producers;
    for (int i = 0; i < 10; ++i) {
        producers.emplace_back(producer);
    }
    std::vector<std::thread> consumers;
    for (int i = 0; i < 3; ++i) {
        consumers.emplace_back(consumer);
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "hello" << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(10));


    return EXIT_SUCCESS;
}
