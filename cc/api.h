#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ops.h"
#include "tensor.h"

namespace peachygrad {

Tensor tensor(pybind11::list py_list, DType dtype);

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
