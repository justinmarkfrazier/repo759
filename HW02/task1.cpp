#include <iostream>
#include <random>
#include <chrono>
#include <ratio>
#include "scan.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

float rand_num()
{
    static thread_local std::mt19937 rng{std::random_device{}()};
    static thread_local std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return dist(rng);
}

int main(int argc, char *argv[])
{
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    int n = std::atoi(argv[1]);

    std::vector<float> array(n), output(n);

    for(int i{0}; i < n; ++i){
        array[i] = rand_num();
    }

    start = high_resolution_clock::now();
    scan(array.data(), output.data(), array.size());
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << duration_sec.count() << "\n";
    std::cout << output[0] << "\n";
    std::cout << output[n-1] << "\n";

    return 0;
}