#include "kernels.h"

#include <immintrin.h>

#include "base.h"

namespace peachygrad {
namespace {

#define FVEC_KERNEL(dst, x, y, mm_size, vec_ub, vec_inst) \
  for (size_t i = 0; i < (vec_ub); i += mm_size) {        \
    __m256 x_vec = _mm256_load_ps((x) + i);               \
    __m256 y_vec = _mm256_load_ps((y) + i);               \
    __m256 vec_dst = (vec_inst)(x_vec, y_vec);            \
    _mm256_store_ps((dst) + i, vec_dst);                  \
  }

#define IVEC_KERNEL(dst, x, y, mm_size, vec_ub, vec_inst)   \
  for (size_t i = 0; i < (vec_ub); i += mm_size) {          \
    __m256i x_vec = _mm256_load_si256((__m256i*)((x) + i)); \
    __m256i y_vec = _mm256_load_si256((__m256i*)((y) + i)); \
    __m256i vec_dst = (vec_inst)(x_vec, y_vec);             \
    _mm256_store_si256((__m256i*)((dst) + i), vec_dst);     \
  }

#define TILE_LEN 64

bool is_matmul_aligned(const Tensor& dst, const Tensor& m0, const Tensor& m1) {
  static constexpr size_t kAlign = sizeof(__m256) / sizeof(float);
  auto divides = [](size_t d) { return d % kAlign == 0; };

  for (int i = 0; i < dst.shape().num_dims; ++i) {
    if (!divides(dst.shape()[i])) return false;
  }

  for (int i = 0; i < m0.shape().num_dims; ++i) {
    if (!divides(m0.shape()[i])) return false;
  }

  for (int i = 0; i < m1.shape().num_dims; ++i) {
    if (!divides(m1.shape()[i])) return false;
  }

  return true;
}

void fmatmul_kernel_unaligned(void* dst, void* m0, void* m1,
                              const size_t dst_size, const Shape& dst_shape,
                              const Shape& m0_shape, const Shape& m1_shape) {
  float* fdst = static_cast<float*>(dst);
  float* fm0 = static_cast<float*>(m0);
  float* fm1 = static_cast<float*>(m1);

  int n = m0_shape[0];
  int d = m0_shape[1];
  int m = m1_shape[1];

  // For each row in `dst`, iterate through the row `d` times, each time loading
  // and element of `m0`, then adding, to each element in the row, the `m0`
  // element multiplied with an element in `m1` in the same column as `dst`.
  // This iteration pattern ensures we always traverse the matrices row-wise,
  // maximizing spatial locality.
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < d; ++k) {
      const int m0_off = i * d;
      const int m1_off = k * m;
      const int dst_off = i * m;
      float m0_elem = fm0[m0_off + k];

      for (int j = 0; j < m; ++j) {
        fdst[dst_off + j] += m0_elem * fm1[m1_off + j];
      }
    }
  }
}

// ---- Matmul Helpers ---- //
// Multiply (n, d), (d, m) -> (n, m).
void fmatmul_kernel_aligned(void* dst, void* m0, void* m1,
                            const size_t dst_size, const Shape& dst_shape,
                            const Shape& m0_shape, const Shape& m1_shape) {
  float* fdst = static_cast<float*>(dst);
  float* fm0 = static_cast<float*>(m0);
  float* fm1 = static_cast<float*>(m1);

  int n = m0_shape[0];
  int d = m0_shape[1];
  int m = m1_shape[1];

  // See `fmatmul_kernel_unaligned` for iteration pattern.
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < d; ++k) {
      const int m0_off = i * d;
      const int m1_off = k * m;
      const int dst_off = i * m;
      float m0_elem = fm0[m0_off + k];
      __m256 m0_vec = _mm256_set1_ps(m0_elem);

      size_t mm_size = sizeof(__m256) / sizeof(int);
      size_t vec_ub = (m / mm_size) * mm_size;
      for (int j = 0; j < vec_ub; j += mm_size) {
        float* dst_j = &fdst[dst_off + j];
        __m256 m1_vec = _mm256_load_ps(&fm1[m1_off + j]);
        __m256 dst_vec = _mm256_load_ps(dst_j);
        __m256 res = _mm256_fmadd_ps(m0_vec, m1_vec, dst_vec);
        _mm256_store_ps(dst_j, res);
      }

      for (int j = vec_ub; j < m; ++j) {
        fdst[dst_off + j] += m0_elem * fm1[m1_off + j];
      }
    }
  }
}

void imatmul_kernel(void* dst, void* m0, void* m1, const size_t dst_size,
                    const Shape& dst_shape, const Shape& m0_shape,
                    const Shape& m1_shape) {
  int* idst = static_cast<int*>(dst);
  int* im0 = static_cast<int*>(m0);
  int* im1 = static_cast<int*>(m1);

  int n = m0_shape[0];
  int d = m0_shape[1];
  int m = m1_shape[1];

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      for (int k = 0; k < d; ++k) {
        idst[i * m + j] += im0[i * d + k] * im1[k * m + j];
      }
    }
  }
}

#undef TILE_LEN

inline __m256 vec_exp(__m256 x_vec) {
  // Shamelessly ripped from https://stackoverflow.com/a/50425370.
  // Black magic. The first answer kind of makes sense, but I am still working
  // through what the mantissa/float representation manipulation means.
  static constexpr int kC0 = 3537;
  static constexpr int kC1 = 13668;
  static constexpr int kC2 = 15817;
  static constexpr int kC3 = -80470;
  static constexpr float kMagic = 12102203.0f;  // 1 << 23 / ln(2)

  __m256i c0 = _mm256_set1_epi32(kC0);
  __m256i c1 = _mm256_set1_epi32(kC1);
  __m256i c2 = _mm256_set1_epi32(kC2);
  __m256i c3 = _mm256_set1_epi32(kC3);
  __m256 magic = _mm256_set1_ps(kMagic);

  __m256i x_vec_int;
  x_vec = _mm256_mul_ps(magic, x_vec);
  x_vec_int = _mm256_cvtps_epi32(x_vec);
  x_vec_int = _mm256_add_epi32(x_vec_int, _mm256_set1_epi32(127 * (1 << 23)));

  __m256i mantissas;
  mantissas = _mm256_sra_epi32(x_vec_int, _mm_cvtsi32_si128(7));
  mantissas = _mm256_and_si256(mantissas, _mm256_set1_epi32(0xffff));

  // quartic spline interpolation. how do you find these values?
  __m256i p = _mm256_mullo_epi32(c0, mantissas);   // c0 * m
  p = _mm256_sra_epi32(p, _mm_cvtsi32_si128(16));  // (c0 * m) >> 16
  p = _mm256_add_epi32(p, c1);                     // (m(c0) >> 16) + c1
  p = _mm256_mullo_epi32(p, mantissas);            // m((m(c0) >> 16) + c1)
  p = _mm256_sra_epi32(p,
                       _mm_cvtsi32_si128(18));  // m((m(c0) >> 16) + c1) >> 18
  p = _mm256_add_epi32(p, c2);  // (m((m(c0) >> 16) + c1) >> 18) + c2
  p = _mm256_mullo_epi32(p,
                         mantissas);  // m((m((m(c0) >> 16) + c1) >> 18) + c2)
  p = _mm256_sra_epi32(
      p,
      _mm_cvtsi32_si128(14));  // m((m((m(c0) >> 16) + c1) >> 18) + c2) >> 14
  p = _mm256_add_epi32(p,
                       c3);  // m((m((m(c0) >> 16) + c1) >> 18) + c2) >> 14 + c3
  p = _mm256_mullo_epi32(p, mantissas);
  p = _mm256_sra_epi32(p, _mm_cvtsi32_si128(11));
  x_vec_int = _mm256_add_epi32(x_vec_int, p);

  return _mm256_castsi256_ps(x_vec_int);
}

inline float scalar_exp(float x) {
  union Bits {
    float f;
    int i;
  };

  static constexpr int kC0 = 3537;
  static constexpr int kC1 = 13668;
  static constexpr int kC2 = 15817;
  static constexpr int kC3 = -80470;
  static constexpr float kMagic = 12102203.0f;  // 1 << 23 / ln(2)

  Bits b;
  b.i = (int32_t)(kMagic * x) + 127 * (1 << 23);
  int32_t m = (b.i >> 7) & 0xffff;  // copy mantissa
  int32_t p = ((kC0 * m) >> 16) + kC1;
  p = ((m * p) >> 18) + kC2;
  p = ((m * p) >> 14) + kC3;
  p = (m * p) >> 11;

  b.i += p;
  return b.f;
}

inline __m256 vec_log(__m256 x_vec) {
  // https://stackoverflow.com/a/39822314
  static constexpr int kMagic = 0x3f2aaaab;
  static constexpr float k1pNeg23 = 1.19209290e-7f;
  static constexpr float kMC0 = 0.230836749f;
  static constexpr float kAC0 = -0.279208571f;
  static constexpr float kMC1 = 0.331826031f;
  static constexpr float kAC1 = -0.498910338f;
  static constexpr float kLog2 = 0.693147182f;

  __m256i x_vec_int = _mm256_castps_si256(x_vec);
  __m256i e = _mm256_sub_epi32(x_vec_int, _mm256_set1_epi32(kMagic));
  e = _mm256_and_si256(e, _mm256_set1_epi32(0xff800000));
  __m256 m = _mm256_castsi256_ps(_mm256_sub_epi32(x_vec_int, e));
  __m256 i = _mm256_mul_ps(_mm256_cvtepi32_ps(e), _mm256_set1_ps(k1pNeg23));
  __m256 f = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
  __m256 s = _mm256_mul_ps(f, f);
  __m256 r = _mm256_fmadd_ps(_mm256_set1_ps(kMC0), f, _mm256_set1_ps(kAC0));
  __m256 t = _mm256_fmadd_ps(_mm256_set1_ps(kMC1), f, _mm256_set1_ps(kAC1));
  r = _mm256_fmadd_ps(r, s, t);
  r = _mm256_fmadd_ps(r, s, f);
  r = _mm256_fmadd_ps(i, _mm256_set1_ps(kLog2), r);
  return r;
}

inline float scalar_log(float x) {
  // https://stackoverflow.com/a/39822314
  union Bits {
    int i;
    float f;
  };
  float r, s, t, i, f;
  Bits m;
  Bits e;
  Bits x_bits;
  x_bits.f = x;
  e.f = x;

  e.i = (e.i - 0x3f2aaaab) & 0xff800000;
  m.i = x_bits.i - e.i;
  i = (float)(e.i) * 1.19209290e-7f;  // 0x1.0p-23
  /* m in [2/3, 4/3] */
  f = m.f - 1.0f;
  s = f * f;
  /* Compute log1p(f) for f in [-1/3, 1/3] */
  r = fmaf(0.230836749f, f, -0.279208571f);  // 0x1.d8c0f0p-3, -0x1.1de8dap-2
  t = fmaf(0.331826031f, f, -0.498910338f);  // 0x1.53ca34p-2, -0x1.fee25ap-2
  r = fmaf(r, s, t);
  r = fmaf(r, s, f);
  r = fmaf(i, 0.693147182f, r);  // 0x1.62e430p-1 // log(2)
  return r;
}
}  // namespace

// ---- Neg ---- //
void fneg(void* dst, void* x, size_t n) {
  float* fdst = static_cast<float*>(dst);
  float* fx = static_cast<float*>(x);

  size_t mm_size = sizeof(__m256) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  for (size_t i = 0; i < (vec_ub); i += mm_size) {
    __m256 x_vec = _mm256_load_ps(fx + i);
    __m256 vec_dst = _mm256_sub_ps(_mm256_setzero_ps(), x_vec);
    _mm256_store_ps(fdst + i, vec_dst);
  }

  // finish stragglers.
  for (size_t i = vec_ub; i < n; ++i) {
    fdst[i] = -fx[i];
  }
}

void ineg(void* dst, void* x, size_t n) {
  int* idst = static_cast<int*>(dst);
  int* ix = static_cast<int*>(x);

  size_t mm_size = sizeof(__m256) / sizeof(int);
  size_t vec_ub = (n / mm_size) * mm_size;
  for (size_t i = 0; i < (vec_ub); i += mm_size) {
    __m256i x_vec = _mm256_load_si256((__m256i*)(ix + i));
    __m256i vec_dst = _mm256_sub_epi32(_mm256_setzero_si256(), x_vec);
    _mm256_store_si256((__m256i*)(idst + i), vec_dst);
  }

  // finish stragglers.
  for (size_t i = vec_ub; i < n; ++i) {
    idst[i] = -ix[i];
  }
}

// ---- Exp ---- //
void fexp(void* dst, void* x, size_t n) {
  float* fdst = static_cast<float*>(dst);
  float* fx = static_cast<float*>(x);

  size_t mm_size = sizeof(__m256) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  for (size_t i = 0; i < (vec_ub); i += mm_size) {
    _mm256_store_ps(fdst + i, vec_exp(_mm256_load_ps(fx + i)));
  }

  // finish stragglers.
  for (size_t i = vec_ub; i < n; ++i) {
    fdst[i] = scalar_exp(fx[i]);
  }
}

void iexp(void* dst, void* x, size_t n) {
  float* fdst = static_cast<float*>(dst);
  int* ix = static_cast<int*>(x);

  size_t mm_size = sizeof(__m256) / sizeof(int);
  size_t vec_ub = (n / mm_size) * mm_size;
  for (size_t i = 0; i < (vec_ub); i += mm_size) {
    _mm256_store_ps(
        fdst + i,
        vec_exp(_mm256_cvtepi32_ps(_mm256_load_si256((__m256i*)(ix + i)))));
  }

  // finish stragglers.
  for (size_t i = vec_ub; i < n; ++i) {
    fdst[i] = scalar_exp((float)(ix[i]));
  }
}

// ---- Log ---- //
void flog(void* dst, void* x, size_t n) {
  float* fdst = static_cast<float*>(dst);
  float* fx = static_cast<float*>(x);

  size_t mm_size = sizeof(__m256) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  for (size_t i = 0; i < (vec_ub); i += mm_size) {
    _mm256_store_ps(fdst + i, vec_log(_mm256_load_ps(fx + i)));
  }

  // finish stragglers.
  for (size_t i = vec_ub; i < n; ++i) {
    fdst[i] = scalar_log(fx[i]);
  }
}

void ilog(void* dst, void* x, size_t n) {
  float* fdst = static_cast<float*>(dst);
  int* ix = static_cast<int*>(x);

  size_t mm_size = sizeof(__m256) / sizeof(int);
  size_t vec_ub = (n / mm_size) * mm_size;
  for (size_t i = 0; i < (vec_ub); i += mm_size) {
    _mm256_store_ps(
        fdst + i,
        vec_log(_mm256_cvtepi32_ps(_mm256_load_si256((__m256i*)(ix + i)))));
  }

  // finish stragglers.
  for (size_t i = vec_ub; i < n; ++i) {
    fdst[i] = scalar_log((float)(ix[i]));
  }
}

// ---- Rcp ---- //
void frcp(void* dst, void* x, size_t n) {
  float* fdst = static_cast<float*>(dst);
  float* fx = static_cast<float*>(x);

  __m256 ones = _mm256_set1_ps(1.0f);
  size_t mm_size = sizeof(__m256) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  for (size_t i = 0; i < vec_ub; i += mm_size) {
    _mm256_store_ps(fdst + i, _mm256_div_ps(ones, _mm256_load_ps(fx + i)));
  }

  // finish stragglers.
  for (size_t i = vec_ub; i < n; ++i) {
    fdst[i] = 1.0f / fx[i];
  }
}

void ircp(void* dst, void* x, size_t n) {
  float* fdst = static_cast<float*>(dst);
  int* ix = static_cast<int*>(x);

  __m256 ones = _mm256_set1_ps(1.0f);
  size_t mm_size = sizeof(__m256) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  for (size_t i = 0; i < vec_ub; i += mm_size) {
    _mm256_store_ps(
        fdst + i,
        _mm256_div_ps(
            ones, _mm256_cvtepi32_ps(_mm256_load_si256((__m256i*)(ix + i)))));
  }

  // finish stragglers.
  for (size_t i = vec_ub; i < n; ++i) {
    fdst[i] = 1.0f / ix[i];
  }
}

// ---- Max ---- //
void fmax(void* dst, void* x, float c, size_t n) {
  float* fdst = static_cast<float*>(dst);
  float* fx = static_cast<float*>(x);

  __m256 c_vec = _mm256_set1_ps(c);
  size_t mm_size = sizeof(__m256) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  for (size_t i = 0; i < vec_ub; i += mm_size) {
    _mm256_store_ps(fdst + i, _mm256_max_ps(_mm256_load_ps(fx + i), c_vec));
  }

  // finish stragglers.
  for (size_t i = vec_ub; i < n; ++i) {
    fdst[i] = std::max(fx[i], c);
  }
}

void imax(void* dst, void* x, float c, size_t n) {
  int* idst = static_cast<int*>(dst);
  int* ix = static_cast<int*>(x);

  __m256i c_vec = _mm256_set1_epi32((int)c);
  size_t mm_size = sizeof(__m256) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  for (size_t i = 0; i < vec_ub; i += mm_size) {
    _mm256_store_si256(
        (__m256i*)(idst + i),
        _mm256_max_epi32(_mm256_load_si256((__m256i*)(ix + i)), c_vec));
  }

  // finish stragglers.
  for (size_t i = vec_ub; i < n; ++i) {
    idst[i] = std::max(ix[i], (int)c);
  }
}

void fmask_gt(void* dst, void* x, float c, size_t n) {
  float* fdst = static_cast<float*>(dst);
  float* fx = static_cast<float*>(x);

  const __m256 ones = _mm256_set1_ps(1.0f);
  const __m256 c_vec = _mm256_set1_ps(c);
  size_t mm_size = sizeof(__m256) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  for (size_t i = 0; i < vec_ub; i += mm_size) {
    __m256 x_vec = _mm256_load_ps(fx + i);
    __m256 mask = _mm256_cmp_ps(x_vec, c_vec, _CMP_GT_OQ);
    __m256 masked_ones = _mm256_and_ps(ones, mask);
    _mm256_store_ps(fdst + i, masked_ones);
  }

  // finish stragglers.
  for (size_t i = vec_ub; i < n; ++i) {
    fdst[i] = fx[i] > c ? 1.0f : 0.0f;
  }
}

void imask_gt(void* dst, void* x, float c, size_t n) {
  float* fdst = static_cast<float*>(dst);
  int* ix = static_cast<int*>(x);

  const __m256 ones = _mm256_set1_ps(1.0f);
  const __m256i c_vec = _mm256_set1_epi32((int)c);
  size_t mm_size = sizeof(__m256) / sizeof(int);
  size_t vec_ub = (n / mm_size) * mm_size;
  for (size_t i = 0; i < vec_ub; i += mm_size) {
    __m256i x_vec = _mm256_load_si256((__m256i*)(ix + i));
    __m256i mask = _mm256_cmpgt_epi32(x_vec, c_vec);
    __m256 masked_ones = _mm256_and_ps(ones, _mm256_castsi256_ps(mask));
    _mm256_store_ps(fdst + i, masked_ones);
  }

  // finish stragglers.
  for (size_t i = vec_ub; i < n; ++i) {
    fdst[i] = ix[i] > (int)c ? 1.0f : 0.0f;
  }
}

// ---- Add ---- //
void fadd(void* dst, void* x, void* y, size_t n) {
  float* fdst = static_cast<float*>(dst);
  float* fx = static_cast<float*>(x);
  float* fy = static_cast<float*>(y);

  size_t mm_size = sizeof(__m256) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  FVEC_KERNEL(fdst, fx, fy, mm_size, vec_ub, _mm256_add_ps);

  // finish stragglers.
  for (size_t i = vec_ub; i < n; ++i) {
    fdst[i] = fx[i] + fy[i];
  }
}

void iadd(void* dst, void* x, void* y, size_t n) {
  int* idst = static_cast<int*>(dst);
  int* ix = static_cast<int*>(x);
  int* iy = static_cast<int*>(y);

  size_t mm_size = sizeof(__m256i) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  IVEC_KERNEL(idst, ix, iy, mm_size, vec_ub, _mm256_add_epi32);

  // compute scalar section.
  for (size_t i = vec_ub; i < n; ++i) {
    idst[i] = ix[i] + iy[i];
  }
}

// ---- Sub ---- //
void fsub(void* dst, void* x, void* y, size_t n) {
  float* fdst = static_cast<float*>(dst);
  float* fx = static_cast<float*>(x);
  float* fy = static_cast<float*>(y);

  size_t mm_size = sizeof(__m256) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  FVEC_KERNEL(fdst, fx, fy, mm_size, vec_ub, _mm256_sub_ps);

  // compute scalar section.
  for (size_t i = vec_ub; i < n; ++i) {
    fdst[i] = fx[i] - fy[i];
  }
}

void isub(void* dst, void* x, void* y, size_t n) {
  int* idst = static_cast<int*>(dst);
  int* ix = static_cast<int*>(x);
  int* iy = static_cast<int*>(y);

  size_t mm_size = sizeof(__m256i) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  IVEC_KERNEL(idst, ix, iy, mm_size, vec_ub, _mm256_sub_epi32);

  // compute scalar section.
  for (size_t i = vec_ub; i < n; ++i) {
    idst[i] = ix[i] - iy[i];
  }
}

// ---- Mul ---- //
void fmul(void* dst, void* x, void* y, size_t n) {
  float* fdst = static_cast<float*>(dst);
  float* fx = static_cast<float*>(x);
  float* fy = static_cast<float*>(y);

  size_t mm_size = sizeof(__m256) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  FVEC_KERNEL(fdst, fx, fy, mm_size, vec_ub, _mm256_mul_ps);

  // compute scalar section.
  for (size_t i = vec_ub; i < n; ++i) {
    fdst[i] = fx[i] * fy[i];
  }
}

void imul(void* dst, void* x, void* y, size_t n) {
  int* idst = static_cast<int*>(dst);
  int* ix = static_cast<int*>(x);
  int* iy = static_cast<int*>(y);

  size_t mm_size = sizeof(__m256i) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  IVEC_KERNEL(idst, ix, iy, mm_size, vec_ub, _mm256_mullo_epi32);

  // compute scalar section.
  for (size_t i = vec_ub; i < n; ++i) {
    idst[i] = ix[i] * iy[i];
  }
}

// ---- Div ---- //
void fdiv(void* dst, void* x, void* y, size_t n) {
  float* fdst = static_cast<float*>(dst);
  float* fx = static_cast<float*>(x);
  float* fy = static_cast<float*>(y);

  size_t mm_size = sizeof(__m256) / sizeof(float);
  size_t vec_ub = (n / mm_size) * mm_size;
  FVEC_KERNEL(fdst, fx, fy, mm_size, vec_ub, _mm256_div_ps);

  // compute scalar section.
  for (size_t i = vec_ub; i < n; ++i) {
    fdst[i] = fx[i] / fy[i];
  }
}

void idiv(void* dst, void* x, void* y, size_t n) {
  int* idst = static_cast<int*>(dst);
  int* ix = static_cast<int*>(x);
  int* iy = static_cast<int*>(y);

  // no vectorization unfortunately.
  for (size_t i = 0; i < n; ++i) {
    idst[i] = ix[i] / iy[i];
  }
}

// ---- Matmul ---- //
// TODO: https://arxiv.org/pdf/2006.06762.pdf
void fmatmul(Tensor& dst, Tensor& x, Tensor& y) {
  // At this point, we can assume shapes are valid.
  Shape x_shape(x.shape());
  Shape y_shape(y.shape());
  if (x.is_vec()) {
    // y is a matrix.
    x_shape = Shape({1, x.shape()[0]});
  } else if (y.is_vec()) {
    // x is a matrix.
    y_shape = Shape({y.shape()[0], 1});
  }

  if (is_matmul_aligned(dst, x, y)) {
    fmatmul_kernel_aligned(dst.raw(), x.raw(), y.raw(), dst.nelems(),
                           dst.shape(), x.shape(), y.shape());
  } else {
    fmatmul_kernel_unaligned(dst.raw(), x.raw(), y.raw(), dst.nelems(),
                             dst.shape(), x.shape(), y.shape());
  }
}

void imatmul(Tensor& dst, Tensor& x, Tensor& y) {
  if (x.is_vec()) {
    // y is a matrix.
    Shape x_shape({1, x.shape()[0]});
    imatmul_kernel(dst.raw(), x.raw(), y.raw(), dst.nelems(), dst.shape(),
                   x_shape, y.shape());
  } else if (y.is_vec()) {
    // x is a matrix.
    Shape y_shape({y.shape()[0], 1});
    imatmul_kernel(dst.raw(), x.raw(), y.raw(), dst.nelems(), dst.shape(),
                   x.shape(), y_shape);
  } else {
    // both are matrices.
    imatmul_kernel(dst.raw(), x.raw(), y.raw(), dst.nelems(), dst.shape(),
                   x.shape(), y.shape());
  }
}

// ---- Reduce Sum ---- //
void freduce_sum(Tensor& dst, Tensor& x, int axis) {
  auto sum_dst = [&dst, axis](const Shape& index, void* it) {
    Shape dst_index = index;
    dst_index[axis] = 0;
    *((float*)(dst.raw()) + LinOffset(dst_index, dst.shape())) +=
        *((float*)(it));
  };
  GenericIterate(
      x, []() {}, sum_dst, []() {});
}

void ireduce_sum(Tensor& dst, Tensor& x, int axis) {
  auto sum_dst = [&dst, axis](const Shape& index, void* it) {
    Shape dst_index = index;
    dst_index[axis] = 0;
    *((int*)(dst.raw()) + LinOffset(dst_index, dst.shape())) += *((int*)(it));
  };
  GenericIterate(
      x, []() {}, sum_dst, []() {});
}

// ---- Reduce Mean ---- //
void freduce_mean(Tensor& dst, Tensor& x, int axis) {
  float ratio = 1.0f / x.shape()[axis];
  auto mean_dst = [&dst, axis, ratio](const Shape& index, void* it) {
    Shape dst_index = index;
    dst_index[axis] = 0;
    float x = *((float*)(it));
    *((float*)(dst.raw()) + LinOffset(dst_index, dst.shape())) += x * ratio;
  };
  GenericIterate(
      x, []() {}, mean_dst, []() {});
}

void ireduce_mean(Tensor& dst, Tensor& x, int axis) {
  float ratio = 1.0f / x.shape()[axis];
  auto mean_dst = [&dst, axis, ratio](const Shape& index, void* it) {
    Shape dst_index = index;
    dst_index[axis] = 0;
    float x = static_cast<float>(*((int*)(it)));
    *((float*)(dst.raw()) + LinOffset(dst_index, dst.shape())) += x * ratio;
  };
  GenericIterate(
      x, []() {}, mean_dst, []() {});
}

// ---- Broadcast ---- //
void fbroadcast(Tensor& dst, Tensor& x, int axis, size_t dup) {
  auto dup_dst = [&dst, axis, dup](const Shape& index, void* it) {
    Shape dst_index = index;
    for (int i = 0; i < dup; ++i) {
      dst_index[axis] = i;
      *((float*)(dst.raw()) + LinOffset(dst_index, dst.shape())) =
          *((float*)(it));
    }
  };
  GenericIterate(
      x, []() {}, dup_dst, []() {});
}

void ibroadcast(Tensor& dst, Tensor& x, int axis, size_t dup) {
  auto dup_dst = [&dst, axis, dup](const Shape& index, void* it) {
    Shape dst_index = index;
    for (int i = 0; i < dup; ++i) {
      dst_index[axis] = i;
      *((int*)(dst.raw()) + LinOffset(dst_index, dst.shape())) = *((int*)(it));
    }
  };
  GenericIterate(
      x, []() {}, dup_dst, []() {});
}

}  // namespace peachygrad
