#include <iostream>

#include <random>
#include <span>

#include <mpi.h>

void do_work(std::span<int, 2> out, int arg, int divisor) {
    out[0] = arg / divisor;
    out[1] = arg % divisor;
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv); // Initialize the runtime

    int n_nodes = 0;
    int rank = 0;

    // get the pool info
    MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Distribute data/work to each of the nodes
    int arg = 0;
    if (rank == 0) {
        for (int other_node = 1; other_node < n_nodes; other_node++) {
            MPI_Send(&other_node, 1, MPI_INT, other_node, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&arg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Do the work
    std::array < int, 2 > out{};
    std::mt19937 gen(0xABBA ^ (rank + n_nodes));
    int my_divisor = std::uniform_int_distribution(2, 25)(gen);
    do_work(out, arg, my_divisor);

    // Retrieve the results from the other nodes
    if (rank == 0) {
        std::array < int, 2 > other_out{};
        for (int other_node = 1; other_node < n_nodes; other_node++) {
            MPI_Recv(other_out.data(), 2, MPI_INT, other_node, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            out[0] += other_out[0];
            out[1] += other_out[1];
        }
    } else {
        MPI_Send(out.data(), 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    // Report/postprocess/etc
    std::cout << out[0] << " " << out[1] << std::endl;

    MPI_Finalize(); // Close the runtime
    return 0;
}