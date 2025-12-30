#include <algorithm>
#include <cassert>
#include <cblas.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

void matmul_openblas(float *A, float *B, float *C, int n) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A, n, B,
              n, 1.0f, C, n);
}

void matmul_cublas(float *A, float *B, float *C, int n) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, (float[]){1.0f}, B, n,
              A, n, (float[]){1.0f}, C, n);

  cublasDestroy(handle);
}

void init_matrix(float *mat, int n) {
  for (int i = 0; i < n * n; ++i) {
    mat[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
}

template <typename Func>
double measure_time(Func matmul_func, float *A, float *B, float *C, int n,
                    int num_runs) {
  std::vector<double> run_times;

  for (int i = 0; i < num_runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    matmul_func(A, B, C, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    run_times.push_back(duration.count());
  }

  std::sort(run_times.begin(), run_times.end());

  // If there is no true "middle" element, average the two middle elements.
  size_t mid = run_times.size() / 2;
  if (run_times.size() % 2 == 0) {
    return (run_times[mid - 1] + run_times[mid]) / 2.0;
  } else {
    return run_times[mid];
  }
}

int main() {
  const int start_size = 2;
  const int end_size = 1024;
  const int step_size = 2;
  const int num_runs = 100;

  cudaSetDevice(0);

  for (int n = start_size; n <= end_size; n += step_size) {
    float *A_cpu = new float[n * n];
    float *B_cpu = new float[n * n];
    float *C_cpu = new float[n * n];

    float *A_gpu;
    float *B_gpu;
    float *C_gpu;

    cudaMalloc(&A_gpu, n * n * sizeof(float));
    cudaMalloc(&B_gpu, n * n * sizeof(float));
    cudaMalloc(&C_gpu, n * n * sizeof(float));

    init_matrix(A_cpu, n);
    init_matrix(B_cpu, n);

    cudaMemcpy(A_gpu, A_cpu, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B_cpu, n * n * sizeof(float), cudaMemcpyHostToDevice);

    double cpu_time =
        measure_time(matmul_openblas, A_cpu, B_cpu, C_cpu, n, num_runs);
    std::cout << cpu_time << "\n";

    double gpu_time =
        measure_time(matmul_cublas, A_gpu, B_gpu, C_gpu, n, num_runs);
    std::cout << gpu_time << "\n";

    delete[] A_cpu;
    delete[] B_cpu;
    delete[] C_cpu;

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
  }

  return 0;
}
