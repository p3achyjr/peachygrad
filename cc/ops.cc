#include "ops.h"

#include <immintrin.h>

#include "base.h"
#include "kernels.h"

namespace peachygrad {
namespace {

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

Tensor elemwise_add(Tensor& x, Tensor& y) {
  CHECK(x.shape() == y.shape());
  CHECK(x.dtype() == y.dtype());
  Tensor sum = Tensor(x.dtype(), x.shape());
  Dispatch(OpCode::kAdd, x.dtype(), sum, x, y);

  return sum;
}

Tensor elemwise_sub(Tensor& x, Tensor& y) {
  CHECK(x.shape() == y.shape());
  CHECK(x.dtype() == y.dtype());
  Tensor diff = Tensor(x.dtype(), x.shape());
  Dispatch(OpCode::kSub, x.dtype(), diff, x, y);

  return diff;
}

Tensor elemwise_mul(Tensor& x, Tensor& y) {
  CHECK(x.shape() == y.shape());
  CHECK(x.dtype() == y.dtype());
  Tensor prod = Tensor(x.dtype(), x.shape());
  Dispatch(OpCode::kMul, x.dtype(), prod, x, y);

  return prod;
}

Tensor elemwise_div(Tensor& x, Tensor& y) {
  CHECK(x.shape() == y.shape());
  CHECK(x.dtype() == y.dtype());
  Tensor quot = Tensor(x.dtype(), x.shape());
  Dispatch(OpCode::kDiv, x.dtype(), quot, x, y);

  return quot;
}

Tensor matmul(Tensor& x, Tensor& y) {
  Tensor mm = Tensor(x.dtype(), MatmulShape(x, y));
  Dispatch(OpCode::kMatMul, x.dtype(), mm, x, y);

  return mm;
}

}  // namespace peachygrad
