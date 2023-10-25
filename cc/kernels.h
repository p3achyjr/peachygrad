#pragma once
#include "tensor.h"

namespace peachygrad {

void fneg(void* dst, void* x, size_t n);
void ineg(void* dst, void* x, size_t n);

// `dst` is a float* in `iexp`.
void fexp(void* dst, void* x, size_t n);
void iexp(void* dst, void* x, size_t n);

// `dst` is a float* in `ilog`.
void flog(void* dst, void* x, size_t n);
void ilog(void* dst, void* x, size_t n);

// `dst` is a float* in `ircp`.
void frcp(void* dst, void* x, size_t n);
void ircp(void* dst, void* x, size_t n);

void fmax(void* dst, void* x, float c, size_t n);
void imax(void* dst, void* x, float c, size_t n);

void fmask_gt(void* dst, void* x, float c, size_t n);
void imask_gt(void* dst, void* x, float c, size_t n);

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

void freduce_sum(Tensor& dst, Tensor& x, int axis);
void ireduce_sum(Tensor& dst, Tensor& x, int axis);

void freduce_mean(Tensor& dst, Tensor& x, int axis);
void ireduce_mean(Tensor& dst, Tensor& x, int axis);

void fbroadcast(Tensor& dst, Tensor& x, int axis, size_t dup);
void ibroadcast(Tensor& dst, Tensor& x, int axis, size_t dup);

using UnopKernel = void (*)(void*, void*, size_t);
using BinopKernel = void (*)(void*, void*, void*, size_t);
using MatmulKernel = void (*)(Tensor& dst, Tensor& x, Tensor& y);

}  // namespace peachygrad
