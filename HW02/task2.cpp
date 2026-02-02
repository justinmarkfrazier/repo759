#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <ratio>
#include "convolution.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

float rand_num(float min, float max)
{
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

int main(int argc, char *argv[])
{
    // Timing variables
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // Get n and m from command line arguments
    int n = std::atoi(argv[1]);
    int m = std::atoi(argv[2]);

    // Prevent potential overflows
    const std::size_t nn = static_cast<std::size_t>(n);
    const std::size_t mm = static_cast<std::size_t>(m);

    // Initialize image, output, and mask arrays (raw new[])
    float* image  = new float[nn * nn];
    float* output = new float[nn * nn];
    float* mask   = new float[mm * mm];

    // Fill image and mask with random numbers
    for(std::size_t i{0}; i < nn*nn; ++i){
        image[i] = rand_num(-10.0f, 10.0f);
    }
    for(std::size_t i{0}; i < mm*mm; ++i){
        mask[i] = rand_num(-1.0f, 1.0f);
    }    

    // Time the convolution operation
    start = high_resolution_clock::now();
    convolve(image, output, nn, mask, mm);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    // Print timing and some output values
    std::cout << duration_sec.count() << "\n";
    std::cout << output[0] << "\n";
    std::cout << output[nn*nn-1] << "\n";

    delete[] image;
    delete[] output;
    delete[] mask;

    return 0;
}