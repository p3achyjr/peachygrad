#pragma once
#include "tensor.h"

namespace peachygrad {

void fadd(void* dst, void* x, void* y, size_t n);
void iadd(void* dst, void* x, void* y, size_t n);

void fsub(void* dst, void* x, void* y, size_t n);
void isub(void* dst, void* x, void* y, size_t n);

void fmul(void* dst, void* x, void* y, size_t n);
void imul(void* dst, void* x, void* y, size_t n);

void fdiv(void* dst, void* x, void* y, size_t n);
void idiv(void* dst, void* x, void* y, size_t n);

void fmatmul(Tensor& dst, Tensor& x, Tensor& y);
void imatmul(Tensor& dst, Tensor& x, Tensor& y);

using BinopKernel = void (*)(void*, void*, void*, size_t);
using MatmulKernel = void (*)(Tensor& dst, Tensor& x, Tensor& y);

}  // namespace peachygrad
