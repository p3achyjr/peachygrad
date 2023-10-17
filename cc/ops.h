#pragma once

#include "tensor.h"

namespace peachygrad {

enum class OpCode : uint16_t {
  kAdd = 0,
  kSub = 1,
  kMul = 2,
  kDiv = 3,
  kMatMul = 4,
};

Tensor elemwise_add(Tensor& x, Tensor& y);
Tensor elemwise_add(Tensor& dst, Tensor& x, Tensor& y);

Tensor elemwise_sub(Tensor& x, Tensor& y);
Tensor elemwise_sub(Tensor& dst, Tensor& x, Tensor& y);

Tensor elemwise_mul(Tensor& x, Tensor& y);
Tensor elemwise_mul(Tensor& dst, Tensor& x, Tensor& y);

Tensor elemwise_div(Tensor& x, Tensor& y);
Tensor elemwise_div(Tensor& dst, Tensor& x, Tensor& y);

Tensor matmul(Tensor& x, Tensor& y);
Tensor matmul(Tensor& dst, Tensor& x, Tensor& y);

}  // namespace peachygrad
