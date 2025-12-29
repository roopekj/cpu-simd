#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string>

void brighten_naive(uint8_t *data, int w, int h, int c, int brightness) {
  size_t total = w * h * c;
  for (size_t i = 0; i < total; ++i) {
    int v = data[i] + brightness;
    data[i] = (uint8_t)std::clamp(v, 0, 255);
  }
}

void brighten_avx512(uint8_t *data, int w, int h, int c, int brightness) {
  size_t total = w * h * c;
  size_t i = 0;

  // Create the 512 bit vector that will be added to each "batch" of values
  // in the image
  __m512i addV = _mm512_set1_epi8((signed char)brightness);

  // vectorized loop: 64 bytes per iteration
  for (; i + 64 <= total; i += 64) {
    __m512i batch = _mm512_loadu_si512((__m512i *)&data[i]);
    __m512i result = _mm512_adds_epu8(batch, addV);
    _mm512_storeu_si512((__m512i *)&data[i], result);
  }

  // 'total' is how many elements there are in data, 'i' is the index
  // from where incrementing by 512 bits would've gone over the array's size
  size_t remaining = total - i;

  if (remaining > 0) {
    // For example, with remaining == 2
    // 0000 0001 -> 0000 0100 -> 0000 0011
    __mmask16 mask = (1ULL << remaining) - 1;

    // Load the data into a register with the mask, other elements are zeroed
    // out.
    __m512i tail = _mm512_maskz_loadu_epi8(mask, &data[i]);

    // Calculate the saturated add on the data.
    __m512i res = _mm512_adds_epu8(tail, addV);

    // Write the result from the register back into the data array using the
    // same mask, so that we don't overwrite prior values with zeroes.
    _mm512_mask_storeu_epi8(&data[i], mask, res);
  }
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " input_file output_file brightness method\n";
    return 1;
  }

  std::string input = argv[1];
  std::string output = argv[2];
  int brightness = std::stoi(argv[3]);
  std::string method = argv[4];

  // Load image
  int w, h, c;
  uint8_t *img = stbi_load(input.c_str(), &w, &h, &c, 0);
  if (!img) {
    std::cerr << "Failed to load image: " << input << "\n";
    return 1;
  }

  std::cout << "Loaded " << w << "x" << h << " with " << c << " channels.\n";

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10000; ++i) {
    if (method == "naive") {
      brighten_naive(img, w, h, c, brightness);
    } else if (method == "avx512") {
      brighten_avx512(img, w, h, c, brightness);
    } else {
      std::cerr << "Invalid method, expected naive or avx512, got " << method
                << "\n";
      stbi_image_free(img);
      return 1;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  // Write output image
  if (!stbi_write_png(output.c_str(), w, h, c, img, w * c)) {
    std::cerr << "Failed to write image: " << output << "\n";
    stbi_image_free(img);
    return 1;
  }

  std::cout << "Saved: " << output << "\n";

  stbi_image_free(img);
  return 0;
}
