#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

namespace peachygrad {

/*
 * PCG 64-32 implementation
 */
class PRng final {
 public:
  // default constructor will create a PRng object with system time as the
  // initial seed.
  PRng();
  // Constructors create and return a new PRng object, supporting 32, 64, and
  // 128-bit numbers respectively.
  PRng(uint64_t seed);
  ~PRng() = default;
  // Disable Copy
  PRng(PRng const&) = delete;
  PRng& operator=(PRng const&) = delete;

  uint32_t next();

  // UniformRandomBitGenerator Impl.
  using result_type = uint32_t;
  static constexpr result_type min() {
    return std::numeric_limits<result_type>::min();
  }
  static constexpr result_type max() {
    return std::numeric_limits<result_type>::max();
  }
  result_type operator()() { return next(); }

  // private:
  uint64_t state_;
};

inline float Uniform(PRng& prng, float lo, float hi) {
  // Assumes [s(1) exp(8) man(23)] formatting.
  // Hardcode exp = 127 to normalize exp term to 1.
  const float scale = hi - lo;
  const uint32_t rand = prng.next();

  // upper bits are the most random for PCG.
  const uint32_t man = rand >> 9;
  const uint32_t exp = static_cast<uint32_t>(127);
  const uint32_t x = exp << 23 | man;

  float res;
  memcpy(&res, &x, sizeof(x));
  res -= 1.0f;

  // res is now normalized between [0, 1].
  return (res * scale) + lo;
}
}  // namespace peachygrad
