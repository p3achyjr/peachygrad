#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ops.h"
#include "tensor.h"

namespace peachygrad {

Tensor tensor(pybind11::list py_list, DType dtype);

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

// Neg.
inline Tensor neg(Tensor& x) { return elemwise_neg(x); }
inline Tensor neg(Tensor& dst, Tensor& x) { return elemwise_neg(dst, x); }

// Exp.
inline Tensor exp(Tensor& x) { return elemwise_exp(x); }
inline Tensor exp(Tensor& dst, Tensor& x) { return elemwise_exp(x); }

// Log.
inline Tensor log(Tensor& x) { return elemwise_log(x); }
inline Tensor log(Tensor& dst, Tensor& x) { return elemwise_log(x); }

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

}  // namespace peachygrad
