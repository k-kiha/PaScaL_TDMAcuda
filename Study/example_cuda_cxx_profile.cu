#include "pascal_tdma_cuda.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int parse_positive_int(const int argc,
                       char** argv,
                       const int index,
                       const int default_value,
                       const char* name) {
    if (argc <= index) {
        return default_value;
    }
    char* end = nullptr;
    const long value = std::strtol(argv[index], &end, 10);
    if (end == argv[index] || *end != '\0' || value <= 0) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return static_cast<int>(value);
}

const char* mpi_mode_name(const pascal_tdma::MpiBufferMode mode) {
    return mode == pascal_tdma::MpiBufferMode::HostStaging ? "host" : "device";
}

void initialize_coefficients(std::vector<double>& h_a,
                             std::vector<double>& h_b,
                             std::vector<double>& h_c,
                             std::vector<double>& h_d,
                             const int nsys,
                             const int nrow,
                             const int rank,
                             const int nprocs) {
    std::fill(h_a.begin(), h_a.end(), 1.0);
    std::fill(h_b.begin(), h_b.end(), -2.0);
    std::fill(h_c.begin(), h_c.end(), 1.0);
    std::fill(h_d.begin(), h_d.end(), 0.0);

    if (rank == 0) {
        for (int sys = 0; sys < nsys; ++sys) {
            h_d[pascal_tdma::index2(sys, 0, nsys)] = -1.0;
        }
    }
    if (rank == nprocs - 1) {
        for (int sys = 0; sys < nsys; ++sys) {
            h_d[pascal_tdma::index2(sys, nrow - 1, nsys)] = -1.0;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    try {
        const int n1 = parse_positive_int(argc, argv, 1, 64, "n1");
        const int n2 = parse_positive_int(argc, argv, 2, 64, "n2");
        const int n3 = parse_positive_int(argc, argv, 3, 2048, "n3");
        const int iterations = parse_positive_int(argc, argv, 4, 10, "iterations");
        const int tdma_threads = parse_positive_int(argc, argv, 5, 128, "tdma_threads");
        const int reduced_threads = parse_positive_int(argc, argv, 6, 128, "reduced_threads");
        if (argc > 7) {
            throw std::invalid_argument(
                "usage: example_cuda_cxx_profile [n1] [n2] [n3] [iterations] [tdma_threads] [reduced_threads]");
        }

        int device_count = 0;
        PASCAL_TDMA_CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            throw std::runtime_error("No CUDA device is visible to this MPI rank.");
        }
        PASCAL_TDMA_CUDA_CHECK(cudaSetDevice(rank % device_count));
        PASCAL_TDMA_CUDA_CHECK(cudaDeviceSynchronize());

        int z_first = 0;
        int z_last = -1;
        pascal_tdma::partition_1d(0, n3 - 1, nprocs, rank, z_first, z_last);
        const int nrow = z_last - z_first + 1;
        const int nsys = n1 * n2;
        const std::size_t nvalues = static_cast<std::size_t>(nsys) * nrow;

        int nrow_min = 0;
        int nrow_max = 0;
        MPI_Reduce(&nrow, &nrow_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&nrow, &nrow_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        std::vector<double> h_a(nvalues);
        std::vector<double> h_b(nvalues);
        std::vector<double> h_c(nvalues);
        std::vector<double> h_d(nvalues);
        initialize_coefficients(h_a, h_b, h_c, h_d, nsys, nrow, rank, nprocs);

        double* d_a = nullptr;
        double* d_b = nullptr;
        double* d_c = nullptr;
        double* d_d = nullptr;
        PASCAL_TDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), nvalues * sizeof(double)));
        PASCAL_TDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), nvalues * sizeof(double)));
        PASCAL_TDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), nvalues * sizeof(double)));
        PASCAL_TDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_d), nvalues * sizeof(double)));

        PASCAL_TDMA_CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), nvalues * sizeof(double), cudaMemcpyHostToDevice));
        PASCAL_TDMA_CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), nvalues * sizeof(double), cudaMemcpyHostToDevice));
        PASCAL_TDMA_CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), nvalues * sizeof(double), cudaMemcpyHostToDevice));
        PASCAL_TDMA_CUDA_CHECK(cudaMemcpy(d_d, h_d.data(), nvalues * sizeof(double), cudaMemcpyHostToDevice));

        const pascal_tdma::MpiBufferMode mpi_mode = pascal_tdma::mpi_mode_from_env();
        pascal_tdma::PascalTdmaPlan plan;
        plan.create(nsys, MPI_COMM_WORLD, tdma_threads, reduced_threads, mpi_mode);

        if (rank == 0) {
            std::cout << "solver,implementation,nranks,n1,n2,n3,nsys,nrow_min,nrow_max,"
                      << "iter,iterations,mpi_mode,total_s_max,total_s_avg\n";
        }

        for (int iter = 0; iter < iterations; ++iter) {
            MPI_Barrier(MPI_COMM_WORLD);

            pascal_tdma::SolveTimings timings;
            pascal_tdma::solve_profiled(plan, d_a, d_b, d_c, d_d, nsys, nrow, &timings);

            const double local_total = timings.total;
            double max_total = 0.0;
            double sum_total = 0.0;
            MPI_Reduce(&local_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_total, &sum_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                std::cout << std::setprecision(12)
                          << "tdma,cuda-cxx"
                          << ',' << nprocs
                          << ',' << n1
                          << ',' << n2
                          << ',' << n3
                          << ',' << nsys
                          << ',' << nrow_min
                          << ',' << nrow_max
                          << ',' << iter
                          << ',' << iterations
                          << ',' << mpi_mode_name(mpi_mode)
                          << ',' << max_total
                          << ',' << (sum_total / nprocs)
                          << '\n';
            }
        }

        plan.destroy();
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFree(d_d);
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " error: " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
