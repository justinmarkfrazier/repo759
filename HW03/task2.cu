#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <random>

int rand_int(int min, int max)
{
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> dist(min, max); 
    return dist(rng);
}


__global__ void simpleKernel(int* data, int a) { 

    int x = threadIdx.x;
    int y = blockIdx.x;

    data[y * blockDim.x + x] = a*x + y;
}

int main() {
    const int numThreads = 8;
    const int numBlocks = 2;
    int hA[numThreads * numBlocks], *dA;

    int a = rand_int(1, 10);
    // allocate memory on the device (GPU); zero out all entries in this device array
    cudaMalloc((void**)&dA, sizeof(int) * numThreads * numBlocks);
    cudaMemset(dA, 0, numThreads * numBlocks * sizeof(int));
    // invoke GPU kernel, with one block and numThreads threads
    simpleKernel<<<numBlocks, numThreads>>>(dA, a);
    // wait for the GPU to finish before accessing on host (CPU)
    cudaDeviceSynchronize();
    // copy the results back to the host (CPU)
    cudaMemcpy(hA, dA, sizeof(int) * numThreads * numBlocks, cudaMemcpyDeviceToHost);
    // print the results
        for (int i = 0; i < numThreads * numBlocks; ++i) {
            std::cout << hA[i] << " ";
        }
    // free the memory allocated on the device (GPU)
    cudaFree(dA);
    return 0;
}