#include "matmul.cuh"
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

int main(int argc, char* argv[])
{
    // default block_dim = 16 unless provided
    unsigned int block_dim = 16;
    if (argc >= 2) block_dim = std::atoi(argv[1]);

    if (block_dim == 0 || block_dim > 32) {
        std::cerr << "block_dim must be in [1, 32]\n";
        return 1;
    }

    bool all_ok = true;

    // ============================================================
    // TEST 1: n = 3, known A,B => known C
    // A = [1 2 3; 4 5 6; 7 8 9]
    // B = [9 8 7; 6 5 4; 3 2 1]
    // C = A*B =
    // [ 30  24  18
    //   84  69  54
    //  138 114  90 ]
    // ============================================================
    {
        const unsigned int n = 3;
        const int Cexp_i[9] = {30,24,18, 84,69,54, 138,114,90};

        int *Ai, *Bi, *Ci;
        float *Af, *Bf, *Cf;
        double *Ad, *Bd, *Cd;

        cudaMallocManaged(&Ai, n*n*sizeof(int));
        cudaMallocManaged(&Bi, n*n*sizeof(int));
        cudaMallocManaged(&Ci, n*n*sizeof(int));

        cudaMallocManaged(&Af, n*n*sizeof(float));
        cudaMallocManaged(&Bf, n*n*sizeof(float));
        cudaMallocManaged(&Cf, n*n*sizeof(float));

        cudaMallocManaged(&Ad, n*n*sizeof(double));
        cudaMallocManaged(&Bd, n*n*sizeof(double));
        cudaMallocManaged(&Cd, n*n*sizeof(double));

        // fill A, B, and zero C
        for (unsigned int r = 0; r < n; ++r) {
            for (unsigned int c = 0; c < n; ++c) {
                unsigned int idx = r*n + c;

                int a = (int)(idx + 1);                 // 1..9
                int b = (int)(9 - idx);                 // 9..1

                Ai[idx] = a;  Bi[idx] = b;  Ci[idx] = 0;
                Af[idx] = (float)a; Bf[idx] = (float)b; Cf[idx] = 0.0f;
                Ad[idx] = (double)a; Bd[idx] = (double)b; Cd[idx] = 0.0;
            }
        }

        matmul_1(Ai, Bi, Ci, n, block_dim);
        matmul_2(Af, Bf, Cf, n, block_dim);
        matmul_3(Ad, Bd, Cd, n, block_dim);

        // check
        bool ok_i = true, ok_f = true, ok_d = true;
        for (unsigned int i = 0; i < n*n; ++i) {
            if (Ci[i] != Cexp_i[i]) ok_i = false;

            float ef = (float)Cexp_i[i];
            double ed = (double)Cexp_i[i];

            if (std::fabs(Cf[i] - ef) > 1e-4f) ok_f = false;
            if (std::fabs(Cd[i] - ed) > 1e-9)  ok_d = false;
        }

        std::cout << "TEST1 (n=3 known): "
                  << "int=" << (ok_i ? "PASS" : "FAIL") << " "
                  << "float=" << (ok_f ? "PASS" : "FAIL") << " "
                  << "double=" << (ok_d ? "PASS" : "FAIL") << "\n";

        all_ok = all_ok && ok_i && ok_f && ok_d;

        cudaFree(Ai); cudaFree(Bi); cudaFree(Ci);
        cudaFree(Af); cudaFree(Bf); cudaFree(Cf);
        cudaFree(Ad); cudaFree(Bd); cudaFree(Cd);
    }

    // ============================================================
    // TEST 2: n = 33, A=all ones, B=all ones => C=all n (=33)
    // This tests non-multiple-of-block_dim behavior when block_dim=16/32, etc.
    // ============================================================
    {
        const unsigned int n = 33;
        const int expected_int = (int)n;
        const float expected_float = (float)n;
        const double expected_double = (double)n;

        int *Ai, *Bi, *Ci;
        float *Af, *Bf, *Cf;
        double *Ad, *Bd, *Cd;

        cudaMallocManaged(&Ai, n*n*sizeof(int));
        cudaMallocManaged(&Bi, n*n*sizeof(int));
        cudaMallocManaged(&Ci, n*n*sizeof(int));

        cudaMallocManaged(&Af, n*n*sizeof(float));
        cudaMallocManaged(&Bf, n*n*sizeof(float));
        cudaMallocManaged(&Cf, n*n*sizeof(float));

        cudaMallocManaged(&Ad, n*n*sizeof(double));
        cudaMallocManaged(&Bd, n*n*sizeof(double));
        cudaMallocManaged(&Cd, n*n*sizeof(double));

        for (unsigned int i = 0; i < n*n; ++i) {
            Ai[i] = 1; Bi[i] = 1; Ci[i] = 0;
            Af[i] = 1.0f; Bf[i] = 1.0f; Cf[i] = 0.0f;
            Ad[i] = 1.0;  Bd[i] = 1.0;  Cd[i] = 0.0;
        }

        matmul_1(Ai, Bi, Ci, n, block_dim);
        matmul_2(Af, Bf, Cf, n, block_dim);
        matmul_3(Ad, Bd, Cd, n, block_dim);

        bool ok_i = true, ok_f = true, ok_d = true;
        for (unsigned int i = 0; i < n*n; ++i) {
            if (Ci[i] != expected_int) ok_i = false;
            if (std::fabs(Cf[i] - expected_float) > 1e-4f) ok_f = false;
            if (std::fabs(Cd[i] - expected_double) > 1e-9)  ok_d = false;
        }

        std::cout << "TEST2 (n=33 ones): "
                  << "int=" << (ok_i ? "PASS" : "FAIL") << " "
                  << "float=" << (ok_f ? "PASS" : "FAIL") << " "
                  << "double=" << (ok_d ? "PASS" : "FAIL") << "\n";

        all_ok = all_ok && ok_i && ok_f && ok_d;

        cudaFree(Ai); cudaFree(Bi); cudaFree(Ci);
        cudaFree(Af); cudaFree(Bf); cudaFree(Cf);
        cudaFree(Ad); cudaFree(Bd); cudaFree(Cd);
    }

    std::cout << (all_ok ? "ALL TESTS PASS\n" : "SOME TESTS FAILED\n");
    return all_ok ? 0 : 1;
}