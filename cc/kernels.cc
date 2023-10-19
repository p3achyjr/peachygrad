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

// ---- Matmul Helpers ---- //
// Multiply (n, d), (d, m) -> (n, m).
void fmatmul_kernel(void* dst, void* m0, void* m1, const size_t dst_size,
                    const Shape& dst_shape, const Shape& m0_shape,
                    const Shape& m1_shape) {
  float* fdst = static_cast<float*>(dst);
  float* fm0 = static_cast<float*>(m0);
  float* fm1 = static_cast<float*>(m1);

  int n = m0_shape[0];
  int d = m0_shape[1];
  int m = m1_shape[1];

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      for (int k = 0; k < d; ++k) {
        fdst[i * m + j] += fm0[i * d + k] * fm1[k * m + j];
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
  IVEC_KERNEL(idst, ix, iy, mm_size, vec_ub, _mm256_mul_epi32);

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
  if (x.is_vec()) {
    // y is a matrix.
    Shape x_shape({1, x.shape()[0]});
    fmatmul_kernel(dst.raw(), x.raw(), y.raw(), dst.nelems(), dst.shape(),
                   x_shape, y.shape());
  } else if (y.is_vec()) {
    // x is a matrix.
    Shape y_shape({y.shape()[0], 1});
    fmatmul_kernel(dst.raw(), x.raw(), y.raw(), dst.nelems(), dst.shape(),
                   x.shape(), y_shape);
  } else {
    // both are matrices.
    fmatmul_kernel(dst.raw(), x.raw(), y.raw(), dst.nelems(), dst.shape(),
                   x.shape(), y.shape());
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

}  // namespace peachygrad
