Code for SIMD experiments, this repository mainly acts as supplemental material to these two blog posts:
* [The anatomy of AVX-512 intrinsics](https://roopekj.github.io/blog/2025/12/30/the-anatomy-of-avx-512-intrinsics.html)
* [Making CPUs adequate at linear algebra](https://roopekj.github.io/blog/2025/12/30/making-cpus-adequeate-at-linear-algebra.html)

The directory structure is as follows:
* `dot_product`: Program for calculating a dot product with both a naive and an AVX-512 variant.
* `image_processing`: Program for brightening an image read from disk with both a naive and an AVX-512 variant.
* `neural_network`: Program for doing a forward pass of a neural network's linear layer with both a naive and an AVX-512 variant.
* `compare`: Program for comparing and visualizing the performance differences between a system's GPU and CPU at GEMMs of different sizes. Uses OpenBLAS and cuBLAS.
