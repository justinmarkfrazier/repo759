#include <cuda.h>
#include <iostream>

__global__ void simpleKernel(int* data) { 
    // compute a! for threadIdx = a
    int a = threadIdx.x;
    int factorial = 1;
    for (int i = 1; i <= a; ++i) {
        factorial *= i;
    }
    data[a] = factorial;
}

int main() {
  const int numThreads = 8;
  int hA[numThreads], *dA;

  // allocate memory on the device (GPU); zero out all entries in this device array
  cudaMalloc((void**)&dA, sizeof(int) * numThreads);
  cudaMemset(dA, 0, numThreads * sizeof(int));
  // invoke GPU kernel, with one block and numThreads threads
  simpleKernel<<<1, numThreads>>>(dA);
  // wait for the GPU to finish before accessing on host (CPU)
  cudaDeviceSynchronize();
  // copy the results back to the host (CPU)
  cudaMemcpy(hA, dA, sizeof(int) * numThreads, cudaMemcpyDeviceToHost);
  // print the results
    for (int i = 0; i < numThreads; ++i) {
        std::cout << hA[i] << "\n";
    }
  return 0;
}