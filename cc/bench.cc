#include <chrono>

#include "ops.h"
#include "random.h"

using namespace ::peachygrad;

inline Tensor uniform(PRng& prng, Shape shape, float lo, float hi) {
  // must return float.
  Tensor t(DType::kF32, shape);
  auto fill_random = [&prng, lo, hi](const Shape& _, void* it) {
    *static_cast<float*>(it) = Uniform(prng, lo, hi);
  };

  GenericIterate(
      t, []() {}, fill_random, []() {});
  return t;
}

int main(int argc, char** argv) {
  static constexpr int kNumIterations = 1000;

  PRng prng;
  Tensor x = uniform(prng, Shape({128, 784}), -.01, .01);
  Tensor w = uniform(prng, Shape({784, 64}), -.01, .01);
  Tensor dst(DType::kF32, Shape({128, 64}));

  double avg_duration = 0.0;
  for (int i = 0; i < kNumIterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    matmul(dst, x, w);
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count();
    avg_duration =
        ((avg_duration * i) / (i + 1)) + (static_cast<double>(dur) / (i + 1));
  }

  std::cout << "Matmul Average Duration: " << avg_duration / 1000.0 << "us\n";
}
