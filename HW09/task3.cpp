#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = std::atoi(argv[1]);

    std::vector<float> sendbuf(n);
    std::vector<float> recvbuf(n);

    for (int i = 0; i < n; ++i) {
        sendbuf[i] = static_cast<float>(i);
    }

    const int peer = 1 - rank;
    const int tag = 0;

    double start = MPI_Wtime();

    if (rank == 0) {
        MPI_Send(sendbuf.data(), n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD);
        MPI_Recv(recvbuf.data(), n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(recvbuf.data(), n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(sendbuf.data(), n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD);
    }

    double end = MPI_Wtime();
    double local_ms = (end - start) * 1000.0;

    if (rank == 0) {
        double other_ms = 0.0;
        MPI_Recv(&other_ms, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double total_ms = local_ms + other_ms;
        std::cout << total_ms << "\n";
    } else {
        MPI_Send(&local_ms, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}