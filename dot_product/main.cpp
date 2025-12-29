#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <random>

using u32 = std::uint32_t;

u32 ceil_to_multiple(u32 n, u32 base) { return (n + base - 1) / base * base; }

float dot_product_naive(const float *a, const float *b, u32 n) {
  float sum = 0.0f;
  for (u32 i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

float dot_product_avx512(float *a, float *b, u32 n) {
  __m512 sum = _mm512_setzero_ps();

  for (u32 i = 0; i < n; i += 16) {
    __m512 va = _mm512_loadu_ps(&a[i]);
    __m512 vb = _mm512_loadu_ps(&b[i]);
    sum = _mm512_fmadd_ps(va, vb, sum);
  }

  return _mm512_reduce_add_ps(sum);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " method\n";
    return 1;
  }

  std::string method = argv[1];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  const u32 n = 1000000000;

  float *a = new float[n];
  float *b = new float[n];

  const u32 padded_size = ceil_to_multiple(n, 16);
  float *pa = new float[padded_size];
  float *pb = new float[padded_size];

  for (u32 i = 0; i < n; ++i) {
    a[i] = pa[i] = dis(gen);
    b[i] = pb[i] = dis(gen);
  }

  for (u32 i = n; i < padded_size; ++i) {
    pa[i] = 0.0f;
    pb[i] = 0.0f;
  }

  auto start = std::chrono::high_resolution_clock::now();

  float result = -1;
  if (method == "naive") {
    result = dot_product_naive(a, b, padded_size);
  } else if (method == "avx512") {
    result = dot_product_avx512(pa, pb, padded_size);
  } else {
    delete[] a;
    delete[] b;
    delete[] pa;
    delete[] pb;
    std::cerr << "Invalid method, expected naive or avx512, got " << method
              << "\n";
    return 1;
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  // Without this, the compiler just skips the calculations :)
  std::cout << result << std::endl;

  delete[] a;
  delete[] b;
  delete[] pa;
  delete[] pb;

  return 0;
}
