#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels.h"
#include "ops.h"
#include "random.h"
#include "tensor.h"

namespace peachygrad {

Tensor tensor(pybind11::array_t<float, pybind11::array::c_style |
                                           pybind11::array::forcecast>
                  np_arr);

Tensor tensor(pybind11::list py_list, DType dtype);

pybind11::array_t<float> numpy(Tensor& t);

inline Tensor zeros(Shape shape, DType dtype) { return Tensor(dtype, shape); }

inline Tensor ones(Shape shape, DType dtype) {
  Tensor t = Tensor(dtype, shape);
  void* buf = t.raw();
  size_t n = t.nelems();
  switch (dtype) {
    case DType::kF32:
      std::fill(static_cast<float*>(buf), static_cast<float*>(buf) + n, 1);
      break;
    case DType::kI32:
      std::fill(static_cast<int*>(buf), static_cast<int*>(buf) + n, 1);
      break;
    default:
      ABORT("Unknown DType in `ones`.");
  }

  return t;
}

inline Tensor constant(Shape shape, DType dtype, float c) {
  Tensor t = Tensor(dtype, shape);
  void* buf = t.raw();
  size_t n = t.nelems();
  switch (dtype) {
    case DType::kF32:
      std::fill(static_cast<float*>(buf), static_cast<float*>(buf) + n, c);
      break;
    case DType::kI32:
      std::fill(static_cast<int*>(buf), static_cast<int*>(buf) + n,
                static_cast<int>(c));
      break;
    default:
      ABORT("Unknown DType in `constant`.");
  }

  return t;
}

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

// Identity.
inline Tensor identity(Tensor& dst, const Tensor& x) {
  memcpy(dst.raw(), x.raw(), DTypeSize(x.dtype()) * x.nelems());
  return dst;
}

inline Tensor identity(const Tensor& x) {
  Tensor clone(x.dtype(), x.shape());
  return identity(clone, x);
}

// Transpose.
inline Tensor transpose(Tensor& dst, Tensor& x) {
  auto transpose_set = [&dst, &x](const Shape& index, void* it) {
    Shape index_t = index.transpose();
    size_t offset = LinOffset(index_t, dst.shape());
    if (dst.dtype() == DType::kF32) {
      *((float*)(dst.raw()) + offset) = *((float*)(it));
    } else if (dst.dtype() == DType::kI32) {
      *((int*)(dst.raw()) + offset) = *((int*)(it));
    }
  };

  GenericIterate(
      x, []() {}, transpose_set, []() {});
  return dst;
}

inline Tensor transpose(Tensor& x) {
  Tensor x_t(x.dtype(), x.shape().transpose());
  return transpose(x_t, x);
}

// Reduce Sum.
inline Tensor reduce_sum(Tensor& dst, Tensor& x, int axis) {
  if (dst.dtype() == DType::kI32) {
    ireduce_sum(dst, x, axis);
  } else {
    freduce_sum(dst, x, axis);
  }

  return dst;
}

inline Tensor reduce_sum(Tensor& x, int axis) {
  Shape dst_shape = x.shape();
  dst_shape[axis] = 1;
  Tensor dst = Tensor(x.dtype(), dst_shape);
  return reduce_sum(dst, x, axis);
}

// Reduce Mean.
inline Tensor reduce_mean(Tensor& dst, Tensor& x, int axis) {
  if (dst.dtype() == DType::kI32) {
    ireduce_mean(dst, x, axis);
  } else {
    freduce_mean(dst, x, axis);
  }

  return dst;
}

inline Tensor reduce_mean(Tensor& x, int axis) {
  Shape dst_shape = x.shape();
  dst_shape[axis] = 1;
  Tensor dst = Tensor(x.dtype(), dst_shape);
  return reduce_mean(dst, x, axis);
}

// Broadcast.
inline Tensor broadcast(Tensor& dst, Tensor& x, int axis, size_t dup) {
  if (dst.dtype() == DType::kI32) {
    ibroadcast(dst, x, axis, dup);
  } else {
    fbroadcast(dst, x, axis, dup);
  }

  return dst;
}

inline Tensor broadcast(Tensor& x, int axis, size_t dup) {
  Shape dst_shape = x.shape();
  dst_shape[axis] = dup;
  Tensor dst = Tensor(x.dtype(), dst_shape);
  return broadcast(dst, x, axis, dup);
}

// Max.
inline Tensor max(Tensor& dst, Tensor& x, float c) {
  if (dst.dtype() == DType::kI32) {
    imax(dst.raw(), x.raw(), c, dst.nelems());
  } else {
    fmax(dst.raw(), x.raw(), c, dst.nelems());
  }

  return dst;
}

inline Tensor max(Tensor& x, float c) {
  Tensor dst(x.dtype(), x.shape());
  return max(dst, x, c);
}

// Mask Eq.
inline Tensor mask_gt(Tensor& dst, Tensor& x, float c) {
  if (dst.dtype() == DType::kI32) {
    imask_gt(dst.raw(), x.raw(), c, dst.nelems());
  } else {
    fmask_gt(dst.raw(), x.raw(), c, dst.nelems());
  }

  return dst;
}

inline Tensor mask_gt(Tensor& x, float c) {
  Tensor dst(DType::kF32, x.shape());
  return mask_gt(dst, x, c);
}

// Neg.
inline Tensor neg(Tensor& x) { return elemwise_neg(x); }
inline Tensor neg(Tensor& dst, Tensor& x) { return elemwise_neg(dst, x); }

// Exp.
inline Tensor exp(Tensor& x) { return elemwise_exp(x); }
inline Tensor exp(Tensor& dst, Tensor& x) { return elemwise_exp(dst, x); }

// Log.
inline Tensor log(Tensor& x) { return elemwise_log(x); }
inline Tensor log(Tensor& dst, Tensor& x) { return elemwise_log(dst, x); }

// Rcp.
inline Tensor rcp(Tensor& x) { return elemwise_rcp(x); }
inline Tensor rcp(Tensor& dst, Tensor& x) { return elemwise_rcp(dst, x); }

// Add.
inline Tensor add(Tensor& x, Tensor& y) { return elemwise_add(x, y); }
inline Tensor add(Tensor& dst, Tensor& x, Tensor& y) {
  return elemwise_add(dst, x, y);
}

// Sub.
inline Tensor sub(Tensor& x, Tensor& y) { return elemwise_sub(x, y); }
inline Tensor sub(Tensor& dst, Tensor& x, Tensor& y) {
  return elemwise_sub(dst, x, y);
}

// Mul.
inline Tensor mul(Tensor& x, Tensor& y) { return elemwise_mul(x, y); }
inline Tensor mul(Tensor& dst, Tensor& x, Tensor& y) {
  return elemwise_mul(dst, x, y);
}

// Div.
inline Tensor div(Tensor& x, Tensor& y) { return elemwise_div(x, y); }
inline Tensor div(Tensor& dst, Tensor& x, Tensor& y) {
  return elemwise_div(dst, x, y);
}

// Matmul.
inline Tensor mmul(Tensor& x, Tensor& y) { return matmul(x, y); }
inline Tensor mmul(Tensor& dst, Tensor& x, Tensor& y) {
  return matmul(dst, x, y);
}

namespace testing {
bool allclose(Tensor& ref, Tensor& x, float atol, float rtol);
}
}  // namespace peachygrad
