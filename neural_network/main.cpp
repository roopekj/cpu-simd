#include <cassert>
#include <chrono>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <string>

using u32 = std::uint32_t;

template <u32 InDims, u32 OutDims> class LinearLayer {
public:
  static constexpr u32 ceil_to_multiple(u32 n, u32 base) {
    return (n + base - 1) / base * base;
  }

  std::random_device rd;

  // Number of input/output dimensions
  static constexpr u32 input_size = InDims;
  static constexpr u32 output_size = OutDims;
  static constexpr u32 padded_output_size = ceil_to_multiple(OutDims, 16);

  // Initialize weights and biases
  void write_parameters() {
    std::mt19937 gen(this->rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (u32 i = 0; i < padded_output_size; ++i) {
      for (u32 j = 0; j < this->input_size; ++j) {

        if (i >= output_size) {
          // This is a padding weight
          weights[j * output_size + i] = 0.0f;
        } else {
          weights[j * output_size + i] = dis(gen);
        }
      }

      if (i >= output_size) {
        // This is a padding bias
        bias[i] = 0.0f;
      } else {
        bias[i] = dis(gen);
      }
    }
  }

  void forward_naive(float *input, float *output) {
    for (u32 i = 0; i < output_size; ++i) {
      float sum = 0.0f;
      for (u32 j = 0; j < input_size; ++j) {
        sum += input[j] * weights[j * this->output_size + i];
      }
      output[i] = sum + bias[i];
    }
  }

  void forward_avx512(const float *input, float *output) {
    constexpr int widthOutput = padded_output_size / 16;

    __m512 res[widthOutput];
    for (int i = 0; i < widthOutput; i++) {
      res[i] = _mm512_loadu_ps(&bias[i * 16]);
    }

    for (int j = 0; j < this->input_size; ++j) {
      __m512 input_neuron = _mm512_set1_ps(input[j]);

      for (int i = 0; i < widthOutput; ++i) {
        __m512 curr_weights =
            _mm512_loadu_ps(&weights[j * this->output_size + i * 16]);

        res[i] = _mm512_fmadd_ps(input_neuron, curr_weights, res[i]);
      }
    }
    for (int i = 0; i < widthOutput; i++) {
      _mm512_storeu_ps(&output[i * 16], res[i]);
    }
  }
  ~LinearLayer() {
    delete[] weights;
    delete[] bias;
  }

private:
  float *weights = new float[input_size * padded_output_size];
  float *bias = new float[padded_output_size];
};

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " method\n";
    return 1;
  }
  std::string method = argv[1];

  const u32 input_size = 1024;
  const u32 output_size = 512;

  LinearLayer<input_size, output_size> layer;
  layer.write_parameters();

  float *a = new float[input_size];
  float *b = new float[output_size];

  float *pb = new float[layer.padded_output_size];

  for (int i = 0; i < input_size; ++i) {
    a[i] = 1.0f;
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 10000; ++i) {
    if (method == "naive") {
      layer.forward_naive(a, b);
    } else if (method == "avx512") {
      layer.forward_avx512(a, pb);
    } else {
      delete[] a;
      delete[] b;
      delete[] pb;

      std::cerr << "Invalid method, expected naive or avx512, got " << method
                << "\n";
      return 1;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  delete[] a;
  delete[] b;
  delete[] pb;

  return 0;
}
