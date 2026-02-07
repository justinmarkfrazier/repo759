#include "matmul.h"
#include <cstddef>
#include <vector>
#include <iostream>

int main()
{
    const unsigned int n = 3;

    double A[n * n] = {
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9
    };

    double B[n * n] = {
        0.9, 0.8, 0.7,
        0.6, 0.5, 0.4,
        0.3, 0.2, 0.1
    };


    std::vector<double> A_vec = {
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9
    };

    std::vector<double> B_vec = {
        0.9, 0.8, 0.7,
        0.6, 0.5, 0.4,
        0.3, 0.2, 0.1
    };

    double C[n * n] = {0};
    mmul1(A, B, C, n);

    // Print the output matrix C
    std::cout << "Result of mmul1:\n";
    for(unsigned int i = 0; i < n; ++i) {
        for(unsigned int j = 0; j < n; ++j) {
            std::cout << C[i * n + j] << " ";
        }
        std::cout << "\n";
    }

    // Reset C to zero for the next multiplication
    for(unsigned int i = 0; i < n * n; ++i) {
        C[i] = 0;
    }
    mmul2(A, B, C, n);

    // Print the output matrix C
    std::cout << "Result of mmul2:\n";
    for(unsigned int i = 0; i < n; ++i) {
        for(unsigned int j = 0; j < n; ++j) {
            std::cout << C[i * n + j] << " ";
        }
        std::cout << "\n";
    }


    // Reset C to zero for the next multiplication
    for(unsigned int i = 0; i < n * n; ++i) {
        C[i] = 0;
    }
    mmul3(A, B, C, n);

    // Print the output matrix C
    std::cout << "Result of mmul3:\n";
    for(unsigned int i = 0; i < n; ++i) {
        for(unsigned int j = 0; j < n; ++j) {
            std::cout << C[i * n + j] << " ";
        }
        std::cout << "\n";
    }

    // Reset C to zero for the next multiplication
    for(unsigned int i = 0; i < n * n; ++i) {
        C[i] = 0;
    }
    mmul4(A_vec, B_vec, C, n);

    // Print the output matrix C
    std::cout << "Result of mmul4:\n";
    for(unsigned int i = 0; i < n; ++i) {
        for(unsigned int j = 0; j < n; ++j) {
            std::cout << C[i * n + j] << " ";
        }
        std::cout << "\n";
    }



    return 0;
}   

