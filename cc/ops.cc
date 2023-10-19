#include "ops.h"

#include <immintrin.h>

#include "base.h"
#include "kernels.h"

namespace peachygrad {
namespace {

static const std::array<UnopKernel, kNumDTypes> kNegKernels = {fneg, ineg};
static const std::array<BinopKernel, kNumDTypes> kAddKernels = {fadd, iadd};
static const std::array<BinopKernel, kNumDTypes> kSubKernels = {fsub, isub};
static const std::array<BinopKernel, kNumDTypes> kMulKernels = {fmul, imul};
static const std::array<BinopKernel, kNumDTypes> kDivKernels = {fdiv, idiv};
static const std::array<MatmulKernel, kNumDTypes> kMatmulKernels = {fmatmul,
                                                                    imatmul};

int DTypeIndex(DType dtype) {
  switch (dtype) {
    case DType::kF32:
      return 0;
    case DType::kI32:
      return 1;
  }

  ABORT("Unknown DType. (%hu)", static_cast<int16_t>(dtype));
}

bool MatmulCompatible(Tensor& x, Tensor& y) {
  if (!((x.is_matrix() && y.is_matrix()) || (x.is_matrix() && y.is_vec()) ||
        (x.is_vec() && y.is_matrix()))) {
    return false;
  }

  if (x.is_vec() && y.is_matrix()) {
    return x.shape()[0] == y.shape()[0];
  }

  return x.shape()[1] == y.shape()[0];
}

Shape MatmulShape(Tensor& x, Tensor& y) {
  CHECK(MatmulCompatible(x, y));
  if (x.is_matrix() && y.is_vec()) {
    return Shape({x.shape()[0], y.shape()[0]});
  }

  return Shape({x.shape()[0], y.shape()[1]});
}

void DispatchUnop(UnopCode op, DType dtype, Tensor& dst, Tensor& x) {
  switch (op) {
    case UnopCode::kNeg:
      return kNegKernels[DTypeIndex(dtype)](dst.raw(), x.raw(),
                                            dst.buf().nelems());
  }
}

void Dispatch(OpCode op, DType dtype, Tensor& dst, Tensor& x, Tensor& y) {
  switch (op) {
    case OpCode::kAdd:
      return kAddKernels[DTypeIndex(dtype)](dst.raw(), x.raw(), y.raw(),
                                            dst.buf().nelems());
    case OpCode::kSub:
      return kSubKernels[DTypeIndex(dtype)](dst.raw(), x.raw(), y.raw(),
                                            dst.buf().nelems());
    case OpCode::kMul:
      return kMulKernels[DTypeIndex(dtype)](dst.raw(), x.raw(), y.raw(),
                                            dst.buf().nelems());
    case OpCode::kDiv:
      return kDivKernels[DTypeIndex(dtype)](dst.raw(), x.raw(), y.raw(),
                                            dst.buf().nelems());
    case OpCode::kMatMul:
      return kMatmulKernels[DTypeIndex(dtype)](dst, x, y);
  }
}

}  // namespace

Tensor elemwise_neg(Tensor& x) {
  Tensor neg = Tensor(x.dtype(), x.shape());
  return elemwise_neg(neg, x);
}

Tensor elemwise_neg(Tensor& dst, Tensor& x) {
  DispatchUnop(UnopCode::kNeg, x.dtype(), dst, x);
  return dst;
}

Tensor elemwise_add(Tensor& x, Tensor& y) {
  CHECK(x.shape() == y.shape());
  CHECK(x.dtype() == y.dtype());
  Tensor sum = Tensor(x.dtype(), x.shape());
  return elemwise_add(sum, x, y);
}

Tensor elemwise_add(Tensor& dst, Tensor& x, Tensor& y) {
  Dispatch(OpCode::kAdd, x.dtype(), dst, x, y);
  return dst;
}

Tensor elemwise_sub(Tensor& x, Tensor& y) {
  CHECK(x.shape() == y.shape());
  CHECK(x.dtype() == y.dtype());
  Tensor diff = Tensor(x.dtype(), x.shape());
  return elemwise_sub(diff, x, y);
}

Tensor elemwise_sub(Tensor& dst, Tensor& x, Tensor& y) {
  Dispatch(OpCode::kSub, x.dtype(), dst, x, y);
  return dst;
}

Tensor elemwise_mul(Tensor& x, Tensor& y) {
  CHECK(x.shape() == y.shape());
  CHECK(x.dtype() == y.dtype());
  Tensor prod = Tensor(x.dtype(), x.shape());
  return elemwise_mul(prod, x, y);
}

Tensor elemwise_mul(Tensor& dst, Tensor& x, Tensor& y) {
  Dispatch(OpCode::kMul, x.dtype(), dst, x, y);
  return dst;
}

Tensor elemwise_div(Tensor& x, Tensor& y) {
  CHECK(x.shape() == y.shape());
  CHECK(x.dtype() == y.dtype());
  Tensor quot = Tensor(x.dtype(), x.shape());
  return elemwise_div(quot, x, y);
}

Tensor elemwise_div(Tensor& dst, Tensor& x, Tensor& y) {
  Dispatch(OpCode::kDiv, x.dtype(), dst, x, y);
  return dst;
}

Tensor matmul(Tensor& x, Tensor& y) {
  Tensor mm = Tensor(x.dtype(), MatmulShape(x, y));
  return matmul(mm, x, y);
}

Tensor matmul(Tensor& dst, Tensor& x, Tensor& y) {
  Tensor mm = Tensor(x.dtype(), MatmulShape(x, y));
  Dispatch(OpCode::kMatMul, x.dtype(), mm, x, y);

  return mm;
}
}  // namespace peachygrad
